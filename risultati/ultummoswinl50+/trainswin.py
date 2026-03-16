#!/usr/bin/env python
"""
train.py - Training MaskDINO Solar Panels (v5 - Swin-L)
=========================================================
[v5-SwinL] Backbone Swin-L invece di ResNet-50
     MAX_ITER 5000 — best AP rilevato a iter ~2000, inutile andare oltre
     STEPS (3000, 4500) — step decay dopo il picco
     WARMUP 500 iters
     LR 0.0001 ADAMW
     BACKBONE_MULTIPLIER 0.05 — Swin-L pre-trainato, LR backbone bassa
     CLASS_WEIGHT 4.0 — aumenta confidenze classificazione
     USE_CHECKPOINT True — gradient checkpointing per 16GB VRAM
     resume=False — sempre da zero

[v4] Filtraggio inline pannelli border, MASK_FORMAT=bitmask, MASK_WEIGHT=2.0
[v3] Risoluzione 1600px, rotazioni cardinali, no ResizeScale/RandomCrop
[v2] Augmentation colore moderata, flip H+V
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

TRAIN_NAME = "solar_train_swinL"
VALID_NAME = "solar_valid_swinL"
OUTPUT_DIR = os.path.join(BASE_DIR, "output_solar_swinLb")

# ==========================================================================================
# SOGLIE FILTRAGGIO ANNOTAZIONI BORDER
# ==========================================================================================
AR_THRESH   = 0.25
FILL_THRESH = 0.55
MIN_AREA    = 200

# ==========================================================================================
# 1. FILTRAGGIO ANNOTAZIONI BORDER
# ==========================================================================================

def _polygon_area(segmentation):
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
    if not os.path.exists(json_path):
        print(f"❌ JSON non trovato: {json_path}")
        sys.exit(1)

    backup = json_path.replace(".json", "_backup_original.json")
    source = backup if os.path.exists(backup) else json_path

    with open(source, 'r') as f:
        data = json.load(f)

    anns_orig = data.get('annotations', [])
    anns_keep = [a for a in anns_orig if not _is_border_panel(a)]
    n_removed = len(anns_orig) - len(anns_keep)

    if not os.path.exists(backup):
        shutil.copy2(json_path, backup)

    data['annotations'] = anns_keep
    ids_with_ann = set(a['image_id'] for a in anns_keep)
    data['images'] = [img for img in data.get('images', []) if img['id'] in ids_with_ann]

    with open(json_path, 'w') as f:
        json.dump(data, f)

    split = "TRAIN" if "train" in json_path else "VALID"
    pct = n_removed / len(anns_orig) * 100 if anns_orig else 0
    print(f"  [{split}] Annotazioni: {len(anns_orig)} → {len(anns_keep)} "
          f"(rimosse {n_removed} border, {pct:.1f}%)")


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
# 3. MAPPER — PIPELINE CONSERVATIVA PANNELLI
# ==========================================================================================

class CustomTrainMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)

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
# 4. SETUP CONFIGURAZIONE — SWIN-L
# ==========================================================================================

def setup(args=None):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskdino_config(cfg)

    YAML_PATH = os.path.join(
        BASE_DIR, "configs", "coco", "instance-segmentation",
        "swin", "maskdino_swinl_bs16_50ep.yaml"
    )

    if not os.path.exists(YAML_PATH):
        print(f"❌ YAML Swin-L non trovato: {YAML_PATH}")
        sys.exit(1)

    cfg.merge_from_file(YAML_PATH)

    # Pesi pre-trainati Swin-L
    SWIN_WEIGHTS = os.path.join(BASE_DIR, "weights", "maskdino_swinl_pretrained.pth")
    if os.path.exists(SWIN_WEIGHTS):
        cfg.MODEL.WEIGHTS = SWIN_WEIGHTS
        print(f"  ✅ Pesi Swin-L: {SWIN_WEIGHTS}")
    else:
        print(f"  ⚠️  Pesi Swin-L non trovati in {SWIN_WEIGHTS}")
        cfg.MODEL.WEIGHTS = ""

    cfg.MODEL.MASK_ON     = True
    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.INPUT.MIN_SIZE_TRAIN = (640, 704, 768, 832, 896, 960)
    cfg.INPUT.MAX_SIZE_TRAIN = 1600
    cfg.INPUT.MIN_SIZE_TEST  = 800
    cfg.INPUT.MAX_SIZE_TEST  = 1600

    # --- GPU ---
    num_gpus          = torch.cuda.device_count()
    IMAGES_PER_GPU    = 1          # Swin-L pesante → 1 img/GPU su 16GB
    GLOBAL_BATCH_SIZE = num_gpus * IMAGES_PER_GPU
    cfg.SOLVER.IMS_PER_BATCH = GLOBAL_BATCH_SIZE

    cfg.SOLVER.BASE_LR             = 0.0001
    cfg.SOLVER.OPTIMIZER           = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.05
    cfg.SOLVER.WEIGHT_DECAY        = 0.05

    # Best AP a iter ~2000 → fermiamo a 5000 con step decay a 3000/4500
    cfg.SOLVER.MAX_ITER     = 30000   # best AP atteso ~iter 2000
    cfg.SOLVER.STEPS        = (23000, 27000)
    cfg.SOLVER.WARMUP_ITERS = 500

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED    = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE  = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

    cfg.TEST.EVAL_PERIOD         = 500
    cfg.SOLVER.CHECKPOINT_PERIOD = 500

    cfg.DATASETS.TRAIN = (TRAIN_NAME,)
    cfg.DATASETS.TEST  = (VALID_NAME,)
    cfg.MODEL.DEVICE   = "cuda"

    num_classes = len(get_classes_from_json(TRAIN_JSON))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES    = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.MaskDINO.NUM_CLASSES     = num_classes

    try:
        cfg.MODEL.MaskDINO.MASK_WEIGHT  = 2.0
        cfg.MODEL.MaskDINO.CLASS_WEIGHT = 4.0
    except Exception:
        pass

    # Gradient checkpointing — riduce VRAM ~30% per Swin-L su 16GB GPU
    try:
        cfg.MODEL.SWIN.USE_CHECKPOINT = True
    except Exception:
        pass

    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  MaskDINO Solar Panels — Training v5 (Swin-L)")
    print(f"{'='*65}")
    print(f"  Backbone           : Swin Transformer Large (200M param) (200M param)")
    print(f"  GPUs               : {num_gpus}")
    print(f"  Batch size totale  : {GLOBAL_BATCH_SIZE}")
    print(f"  Risoluzione max    : 1600px")
    print(f"  MAX_ITER           : {cfg.SOLVER.MAX_ITER} (best AP atteso ~2000)")
    print(f"  STEPS              : {cfg.SOLVER.STEPS}")
    print(f"  WARMUP             : {cfg.SOLVER.WARMUP_ITERS}")
    print(f"  EVAL_PERIOD        : {cfg.TEST.EVAL_PERIOD}")
    print(f"  Output             : {OUTPUT_DIR}")
    print(f"{'='*65}\n")

    return cfg


# ==========================================================================================
# 5. MAIN
# ==========================================================================================

def main(args=None):
    print("\n🔧 Pre-processing dataset...")

    print("\n📐 Filtraggio annotazioni border:")
    filter_border_annotations(TRAIN_JSON)
    filter_border_annotations(VALID_JSON)

    print("\n🧹 Pulizia annotazioni corrotte:")
    clean_and_verify_dataset(TRAIN_JSON)
    clean_and_verify_dataset(VALID_JSON)

    for name in [TRAIN_NAME, VALID_NAME]:
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
        if name in MetadataCatalog:
            MetadataCatalog.remove(name)

    register_coco_instances(TRAIN_NAME, {}, TRAIN_JSON, TRAIN_IMG)
    register_coco_instances(VALID_NAME, {}, VALID_JSON, VALID_IMG)

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

    # resume=False — parte sempre da zero con pesi Swin-L pre-trainati
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