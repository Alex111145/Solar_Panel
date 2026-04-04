#!/usr/bin/env python
"""
train.py - Training MaskDINO Solar Panels (v4 - FINAL)
=======================================================
Tutte le ottimizzazioni accumulate dalle iterazioni precedenti:

[v2] Risoluzione input 1600px, rotazioni cardinali 0/90/180/270°
[v3] Rimosso ResizeScale e RandomCrop (distruggevano pannelli 30-80px)
     MASK_FORMAT="bitmask" (MaskDINO non supporta "polygon" → crash)
     MASK_WEIGHT=2.0 per penalizzare mask imprecise
     resume=False per partire sempre da zero
[v4] Filtraggio inline pannelli troncati al bordo del tile:
     - aspect ratio < 0.25 → pannello tagliato → rimosso
     - fill ratio polygon/bbox < 0.55 → pannello parziale → rimosso
     - area < 200px² → troppo piccolo → rimosso
     Questo elimina le ~5000 annotazioni che insegnavano al modello
     la forma sbagliata del pannello (sliver 1:10 invece di rettangolo)
"""

import os
import sys
import json
import shutil
import math
import torch
import detectron2.data.transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, launch, HookBase, hooks
from detectron2.utils.events import get_event_storage
from detectron2.data import DatasetMapper
from detectron2.evaluation import COCOEvaluator
from maskdino import add_maskdino_config

# ==========================================================================================
# CONFIGURAZIONE PERCORSI
# ==========================================================================================
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_NAME = "solar_datasets_nuoveannotazioni"
TRAIN_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train", "_annotations.coco.json")
TRAIN_IMG  = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train")
VALID_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json")
VALID_IMG  = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid")

TRAIN_NAME = "solar_train_final"
VALID_NAME = "solar_valid_final"
OUTPUT_DIR = os.path.join(BASE_DIR, "output_solar_nuovor50")

# ==========================================================================================
# SOGLIE FILTRAGGIO ANNOTAZIONI BORDER
# ==========================================================================================
AR_THRESH   = 0.25   # aspect ratio min (w/h e h/w) — sotto = pannello troncato
FILL_THRESH = 0.55   # fill ratio min polygon/bbox  — sotto = pannello parziale
MIN_AREA    = 200    # area minima px²              — sotto = troppo piccolo

# ==========================================================================================
# 1. FILTRAGGIO ANNOTAZIONI BORDER (inline, con backup automatico)
# ==========================================================================================

def _polygon_area(segmentation):
    """Area del poligono con formula di Gauss."""
    if not segmentation or not isinstance(segmentation[0], list):
        return 0
    coords = segmentation[0]
    xs, ys = coords[0::2], coords[1::2]
    n = len(xs)
    if n < 3:
        return 0
    area = sum(xs[i] * ys[(i+1) % n] - xs[(i+1) % n] * ys[i] for i in range(n))
    return abs(area) / 2.0


def _is_border_panel(ann):
    """True se l'annotazione è un pannello troncato al bordo del tile."""
    bbox = ann.get('bbox', [0, 0, 1, 1])
    area = ann.get('area', 0)
    seg  = ann.get('segmentation', [])
    w, h = bbox[2], bbox[3]

    if h <= 0 or w <= 0:
        return True
    if area < MIN_AREA:
        return True
    if w / h < AR_THRESH or h / w < AR_THRESH:
        return True

    poly_a = _polygon_area(seg)
    bbox_a = w * h
    if poly_a > 0 and bbox_a > 0:
        fill = min(poly_a, bbox_a) / max(poly_a, bbox_a)
        if fill < FILL_THRESH:
            return True

    return False


def filter_border_annotations(json_path):
    """
    Rimuove annotazioni di pannelli troncati al bordo del tile.
    Salva backup automatico come _backup_original.json (solo al primo run).
    """
    if not os.path.exists(json_path):
        print(f"❌ JSON non trovato: {json_path}")
        sys.exit(1)

    backup = json_path.replace(".json", "_backup_original.json")

    # Se esiste già un backup significa che abbiamo già filtrato: lavora sul backup
    source = backup if os.path.exists(backup) else json_path

    with open(source, 'r') as f:
        data = json.load(f)

    anns_orig = data.get('annotations', [])
    anns_keep = [a for a in anns_orig if not _is_border_panel(a)]
    n_removed  = len(anns_orig) - len(anns_keep)

    # Backup (solo se non esiste ancora)
    if not os.path.exists(backup):
        shutil.copy2(json_path, backup)

    data['annotations'] = anns_keep

    # Rimuovi immagini senza annotazioni
    ids_with_ann = set(a['image_id'] for a in anns_keep)
    data['images'] = [img for img in data.get('images', []) if img['id'] in ids_with_ann]

    with open(json_path, 'w') as f:
        json.dump(data, f)

    split = "TRAIN" if "train" in json_path else "VALID"
    pct = n_removed / len(anns_orig) * 100 if anns_orig else 0
    print(f"  [{split}] Annotazioni: {len(anns_orig)} → {len(anns_keep)} "
          f"(rimosse {n_removed} pannelli border, {pct:.1f}%)")


def clean_and_verify_dataset(json_path):
    if not os.path.exists(json_path):
        return
    with open(json_path, 'r') as f:
        data = json.load(f)
    initial = len(data['annotations'])
    cleaned = [
        a for a in data['annotations']
        if a.get('area', 0) > 50
        and a['bbox'][2] > 5
        and a['bbox'][3] > 5
        and len(a.get('segmentation', [])) > 0
    ]
    removed = initial - len(cleaned)
    if removed > 0:
        data['annotations'] = cleaned
        with open(json_path, 'w') as f:
            json.dump(data, f)
        print(f"  Pulizia: rimosse {removed} annotazioni corrotte.")


def get_classes_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [c['name'] for c in sorted(data.get('categories', []), key=lambda x: x['id'])]

# ==========================================================================================
# 2. LOGGING JSON LINES
# ==========================================================================================

class CSVLogHook(HookBase):
    """
    Salva le metriche di training in formato JSON Lines (una riga JSON per iterazione).
    Questo formato è robusto al numero variabile di colonne tra iterazioni successive,
    problema che si verifica con csv.DictWriter quando Detectron2 aggiunge dinamicamente
    nuove chiavi (es. loss per singola query del transformer).

    Output: metrics_all.csv  (compatibile con plot_training.py)
    """

    def __init__(self, period=100):
        self._period = period

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self._period != 0:
            return

        storage     = get_event_storage()
        output_file = os.path.join(self.trainer.cfg.OUTPUT_DIR, "metrics_all.csv")
        all_metrics = {"iteration": next_iter}

        for key, history in storage.histories().items():
            try:
                val = history.latest()
                if not (math.isnan(val) or math.isinf(val)):
                    all_metrics[key] = round(float(val), 8)
            except Exception:
                continue

        with open(output_file, mode='a') as f:
            f.write(json.dumps(all_metrics) + "\n")

        if next_iter % 500 == 0:
            loss      = all_metrics.get('total_loss', 0)
            mask_loss = all_metrics.get('loss_mask', 0)
            lr        = all_metrics.get('lr', 0)
            print(f"   --> [Iter {next_iter:5d}] Loss: {loss:.4f} | "
                  f"MaskLoss: {mask_loss:.4f} | LR: {lr:.6f}")

# ==========================================================================================
# 3. MAPPER v4 — PIPELINE PER PANNELLI INTERI (post filtraggio border)
# ==========================================================================================

class CustomTrainMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)

        # Pipeline conservativa per pannelli 30-80px (risoluzione alta + no crop):
        #   ✓ ResizeShortestEdge alta  → pannelli 30-80px diventano 45-130px
        #   ✓ RandomRotation cardinale → pannelli aerei senza orientamento fisso
        #   ✓ Flip H+V                → invarianza a simmetrie ortogonali
        #   ✓ Colore moderato         → robustezza a variazioni luce/stagione/sensore
        #   ✗ ResizeScale             → rimosso: zoomava OUT sotto la soglia small
        #   ✗ RandomCrop              → rimosso: troncava i bordi delle mask

        self.tfm_gens = [
            T.ResizeShortestEdge(
                short_edge_length=(640, 704, 768, 832, 896, 960),
                max_size=1600,
                sample_style="choice"
            ),
            T.RandomRotation(
                angle=[0, 90, 180, 270],
                expand=True,
                sample_style="choice"
            ),
            T.RandomFlip(prob=0.5, horizontal=True,  vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomBrightness(0.75, 1.25),
            T.RandomContrast(0.75, 1.25),
            T.RandomSaturation(0.80, 1.20),
            T.RandomLighting(0.7),
        ]


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=CustomTrainMapper(cfg, is_train=True))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

# ==========================================================================================
# 4. SETUP CONFIGURAZIONE v4
# ==========================================================================================

def setup(args=None):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskdino_config(cfg)

    YAML_PATH = os.path.join(
        BASE_DIR, "configs", "coco", "instance-segmentation",
        "maskdino_R50_bs16_50ep_3s.yaml"
    )
    cfg.merge_from_file(YAML_PATH)

    cfg.MODEL.WEIGHTS     = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
    cfg.MODEL.MASK_ON     = True
    # MaskDINO richiede bitmask obbligatoriamente (gt_masks.tensor interno)
    # PolygonMasks non ha .tensor → crash con "polygon"
    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.INPUT.MIN_SIZE_TRAIN = (640, 704, 768, 832, 896, 960)
    cfg.INPUT.MAX_SIZE_TRAIN = 1600
    cfg.INPUT.MIN_SIZE_TEST  = 800
    cfg.INPUT.MAX_SIZE_TEST  = 1600

    # --- GPU ---
    num_gpus          = torch.cuda.device_count()
    IMAGES_PER_GPU    = 2
    GLOBAL_BATCH_SIZE = num_gpus * IMAGES_PER_GPU
    cfg.SOLVER.IMS_PER_BATCH = GLOBAL_BATCH_SIZE

    cfg.SOLVER.BASE_LR             = 0.0001
    cfg.SOLVER.OPTIMIZER           = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.WEIGHT_DECAY        = 0.05

    cfg.SOLVER.MAX_ITER     = 15000
    cfg.SOLVER.STEPS        = (9000, 12000)
    cfg.SOLVER.WARMUP_ITERS = 1500

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED    = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE  = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

    cfg.TEST.EVAL_PERIOD          = 500
    cfg.SOLVER.CHECKPOINT_PERIOD  = 500

    cfg.DATASETS.TRAIN = (TRAIN_NAME,)
    cfg.DATASETS.TEST  = (VALID_NAME,)
    cfg.MODEL.DEVICE   = "cuda"

    num_classes = len(get_classes_from_json(TRAIN_JSON))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES    = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.MaskDINO.NUM_CLASSES     = num_classes

    # Peso doppio sulla mask loss → penalizza bordi imprecisi durante training
    try:
        cfg.MODEL.MaskDINO.MASK_WEIGHT  = 2.0
        cfg.MODEL.MaskDINO.CLASS_WEIGHT = 4.0
    except Exception:
        pass  # versioni vecchie di MaskDINO non hanno questo parametro

    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MaskDINO Solar Panels — Training v4 (R50)")
    print(f"{'='*60}")
    print(f"  GPUs               : {num_gpus}")
    print(f"  Batch size totale  : {GLOBAL_BATCH_SIZE}")
    print(f"  Risoluzione max    : 1600px")
    print(f"  Mask format        : bitmask (obbligatorio per MaskDINO)")
    print(f"  MAX_ITER           : {cfg.SOLVER.MAX_ITER}")
    print(f"  STEPS              : {cfg.SOLVER.STEPS}")
    print(f"  WARMUP             : {cfg.SOLVER.WARMUP_ITERS}")
    print(f"  EVAL_PERIOD        : {cfg.TEST.EVAL_PERIOD}")
    print(f"  Output             : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    return cfg


# ==========================================================================================
# 5. MAIN
# ==========================================================================================

def main(args=None):
    print("\n🔧 Pre-processing dataset...")

    # Step 1: filtra pannelli troncati al bordo (con backup automatico)
    print("\n📐 Filtraggio annotazioni border (pannelli troncati al bordo tile):")
    filter_border_annotations(TRAIN_JSON)
    filter_border_annotations(VALID_JSON)

    # Step 2: pulizia annotazioni corrotte
    print("\n🧹 Pulizia annotazioni corrotte:")
    clean_and_verify_dataset(TRAIN_JSON)
    clean_and_verify_dataset(VALID_JSON)

    # Step 3: registra dataset (pulizia catalogo per evitare conflitti)
    for name in [TRAIN_NAME, VALID_NAME]:
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
        if name in MetadataCatalog:
            MetadataCatalog.remove(name)

    register_coco_instances(TRAIN_NAME, {}, TRAIN_JSON, TRAIN_IMG)
    register_coco_instances(VALID_NAME, {}, VALID_JSON, VALID_IMG)

    # Step 4: setup e training
    cfg = setup(args)

    with open(os.path.join(OUTPUT_DIR, "config_used.yaml"), "w") as f:
        f.write(cfg.dump())

    trainer = CustomTrainer(cfg)
    trainer.register_hooks([CSVLogHook(period=100)])
    trainer.register_hooks([
        hooks.BestCheckpointer(
            cfg.TEST.EVAL_PERIOD,
            trainer.checkpointer,
            val_metric="segm/AP"
        )
    ])

    # Sempre da zero: le fix cambiano la distribuzione dei dati in modo fondamentale
    trainer.resume_or_load(resume=False)
    return trainer.train()


if __name__ == "__main__":
    gpus_available = torch.cuda.device_count()
    launch(
        main,
        num_gpus_per_machine=gpus_available,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(None,)
    )