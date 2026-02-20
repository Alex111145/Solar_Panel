#!/usr/bin/env python
"""
train_net_v3.py - MaskDINO Solar Panels - BACKBONE SWIN-L
==========================================================
Differenze rispetto a v2 (ResNet-50):

  BACKBONE:
  - ResNet-50  → CNN classica, 25M parametri, vede solo contesto locale
  - Swin-L     → Vision Transformer, 197M parametri, vede contesto globale
                  con "shifted windows" → capisce struttura geometrica ripetuta
                  dei pannelli solari molto meglio

  PERCHÉ SWIN-L È MEGLIO PER IL TUO CASO:
  1. Pannelli regolari e ripetuti  → il Transformer capisce pattern globali
  2. Pannelli parziali ai bordi   → il contesto globale aiuta il riconoscimento
  3. Dataset large (1775 img)     → abbastanza dati per sfruttare i 197M params
  4. Hai GPU potente              → puoi permetterti il costo computazionale

  STIMA IMPATTO:
  - ResNet-50  ottimizzato: ~55-60% mAP@50:95
  - Swin-L                : ~65-75% mAP@50:95

  SETUP INIZIALE (esegui UNA VOLTA prima del training):
    mkdir -p weights
    wget -O weights/maskdino_swinl_pretrained.pth \
      https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth

  USO:
    CUDA_VISIBLE_DEVICES=0,1 python train_net_v3.py
"""

import os
import sys
import json
import csv
import math
import copy
import warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")

import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_train_loader,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer, launch, HookBase, hooks
from detectron2.utils.events import get_event_storage

try:
    from maskdino import add_maskdino_config
except ImportError:
    print("❌ Errore: 'maskdino' non trovato. Lancia da root del progetto.")
    sys.exit(1)

# ==============================================================================
# ⚙️  CONFIGURAZIONE PERCORSI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_FOLDER_NAME = "solar_datasets500"
TRAIN_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train", "_annotations.coco.json")
TRAIN_IMG  = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train")
VALID_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json")
VALID_IMG  = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid")

TRAIN_NAME = "solar_train_v3"
VALID_NAME = "solar_valid_v3"

OUTPUT_DIR = os.path.join(BASE_DIR, "output_solar_swinl")

# ── Config YAML Swin-L ────────────────────────────────────────────────────────
# Il repo MaskDINO include questa config in configs/coco/instance-segmentation/swin/
YAML_PATH = os.path.join(
    BASE_DIR, "configs", "coco", "instance-segmentation",
    "swin", "maskdino_swinl_bs16_50ep.yaml"
)

# ── Pesi pre-addestrati Swin-L ────────────────────────────────────────────────
# Scarica con:
#   mkdir -p weights
#   wget -O weights/maskdino_swinl_pretrained.pth \
#     https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth
SWINL_WEIGHTS = os.path.join(BASE_DIR, "weights", "maskdino_swinl_pretrained.pth")


# ==============================================================================
# 1.  DOWNLOAD AUTOMATICO PESI (se non presenti)
# ==============================================================================

def download_swinl_weights():
    """Scarica i pesi Swin-L pre-addestrati su COCO se non già presenti."""
    if os.path.exists(SWINL_WEIGHTS):
        size_mb = os.path.getsize(SWINL_WEIGHTS) / 1e6
        print(f"✅ Pesi Swin-L trovati: {SWINL_WEIGHTS} ({size_mb:.0f} MB)")
        return

    os.makedirs(os.path.dirname(SWINL_WEIGHTS), exist_ok=True)
    url = (
        "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/"
        "maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_"
        "mask52.3ap_box59.0ap.pth"
    )
    print(f"⬇️  Download pesi Swin-L (~800 MB)...")
    print(f"    URL: {url}")
    print(f"    Dest: {SWINL_WEIGHTS}")

    try:
        import urllib.request
        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / 1e6
            print(f"\r    {pct:.1f}% ({mb:.0f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, SWINL_WEIGHTS, reporthook=progress)
        print(f"\n✅ Download completato!")
    except Exception as e:
        print(f"\n❌ Download fallito: {e}")
        print("   Scarica manualmente con:")
        print(f"   mkdir -p weights && wget -O {SWINL_WEIGHTS} '{url}'")
        sys.exit(1)


# ==============================================================================
# 2.  PULIZIA DATASET
# ==============================================================================

def clean_and_verify_dataset(json_path, min_area=30, min_side=3):
    if not os.path.exists(json_path):
        print(f"⚠️  File non trovato: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    initial = len(data["annotations"])
    cleaned = [
        ann for ann in data["annotations"]
        if ann.get("area", 0) > min_area
        and ann["bbox"][2] > min_side
        and ann["bbox"][3] > min_side
        and len(ann.get("segmentation", [])) > 0
    ]

    removed = initial - len(cleaned)
    if removed > 0:
        data["annotations"] = cleaned
        with open(json_path, "w") as f:
            json.dump(data, f)
        print(f"🧹 {os.path.basename(json_path)}: rimosse {removed} annotazioni sporche")
    else:
        print(f"✅ {os.path.basename(json_path)}: {initial} annotazioni OK")


def get_num_classes(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return len(data.get("categories", []))


# ==============================================================================
# 3.  CUSTOM MAPPER — identico a v2, ottimizzato per immagini aeree
# ==============================================================================

class SolarPanelMapper(DatasetMapper):
    """
    Augmentation ottimizzata per pannelli solari aerei con Swin-L.
    
    Con Swin-L usiamo risoluzioni leggermente più basse rispetto al massimo
    possibile (max 1333 invece di 2000) per bilanciare qualità e VRAM.
    Swin-L con 1333px usa circa 16-20 GB VRAM per immagine.
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.is_train = is_train

        if is_train:
            self.augmentations = T.AugmentationList([
                # ── 1. Multi-scale (ridotto rispetto a v2 per Swin-L + VRAM) ──
                T.ResizeShortestEdge(
                    short_edge_length=(480, 512, 576, 640, 704, 768, 832, 900),
                    max_size=1333,
                    sample_style="choice",
                ),
                # ── 2. Rotazione 90° multipli ─────────────────────────────────
                T.RandomRotation(
                    angle=[0, 90, 180, 270],
                    expand=False,
                    sample_style="choice",
                ),
                # ── 3. Rotazione fine ─────────────────────────────────────────
                T.RandomRotation(
                    angle=[-15, 15],
                    expand=False,
                    sample_style="range",
                ),
                # ── 4. Flip ───────────────────────────────────────────────────
                T.RandomFlip(prob=0.5, horizontal=True,  vertical=False),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                # ── 5. Color jitter ───────────────────────────────────────────
                T.RandomBrightness(0.6, 1.4),
                T.RandomContrast(0.6, 1.4),
                T.RandomSaturation(0.7, 1.3),
                T.RandomLighting(0.7),
                # ── 6. Random crop conservativo ───────────────────────────────
                T.RandomCrop("relative_range", (0.85, 1.0)),
            ])
        else:
            self.augmentations = T.AugmentationList([
                T.ResizeShortestEdge(
                    short_edge_length=800,
                    max_size=1333,
                    sample_style="choice",
                ),
            ])

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        # Compatibilità Detectron2 vecchie/nuove versioni
        img_fmt = getattr(self, "image_format", getattr(self, "img_format", "BGR"))
        image = utils.read_image(dataset_dict["file_name"], format=img_fmt)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)).astype("float32")
        )

        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image.shape[:2]
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image.shape[:2], mask_format="bitmask"
            )
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict


# ==============================================================================
# 4.  CSV LOGGER
# ==============================================================================

class CSVLogHook(HookBase):
    def __init__(self, period=100):
        self._period = period

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self._period != 0:
            return

        storage = get_event_storage()
        output_file = os.path.join(self.trainer.cfg.OUTPUT_DIR, "metrics_all.csv")

        all_metrics = {"iteration": next_iter}
        for key, history in storage.histories().items():
            try:
                val = history.latest()
                all_metrics[key] = 0.0 if (math.isnan(val) or math.isinf(val)) else val
            except Exception:
                continue

        file_exists = os.path.isfile(output_file)
        fieldnames = sorted(all_metrics.keys())

        with open(output_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(all_metrics)

        if next_iter % 500 == 0:
            loss = all_metrics.get("total_loss", 0)
            lr   = all_metrics.get("lr", 0)
            print(f"   [Iter {next_iter:>6}] Loss: {loss:.4f} | LR: {lr:.2e}")


# ==============================================================================
# 5.  CUSTOM TRAINER
# ==============================================================================

class SolarTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=SolarPanelMapper(cfg, is_train=True),
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        from detectron2.evaluation import COCOEvaluator
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


# ==============================================================================
# 6.  SETUP CONFIG SWIN-L
# ==============================================================================

def setup():
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskdino_config(cfg)

    # Verifica esistenza YAML Swin-L
    if not os.path.exists(YAML_PATH):
        print(f"❌ YAML Swin-L non trovato: {YAML_PATH}")
        print("   Struttura attesa:")
        print("   configs/coco/instance-segmentation/swin/maskdino_swinl_bs16_50ep.yaml")
        print("\n   Se il file ha un nome diverso, elenca i file disponibili con:")
        print("   ls configs/coco/instance-segmentation/swin/")
        sys.exit(1)

    cfg.merge_from_file(YAML_PATH)

    # ── Dataset ────────────────────────────────────────────────────────────────
    cfg.DATASETS.TRAIN = (TRAIN_NAME,)
    cfg.DATASETS.TEST  = (VALID_NAME,)

    # ── Modello: carica pesi Swin-L pre-addestrati su COCO ────────────────────
    # Questi pesi già includono il backbone Swin-L addestrato su ImageNet
    # + il decoder MaskDINO addestrato su COCO → transfer learning potente
    cfg.MODEL.WEIGHTS     = SWINL_WEIGHTS
    cfg.MODEL.MASK_ON     = True
    cfg.INPUT.MASK_FORMAT = "bitmask"

    num_classes = get_num_classes(TRAIN_JSON)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES    = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.MaskDINO.NUM_CLASSES     = num_classes

    # ── Device & Batch ─────────────────────────────────────────────────────────
    cfg.MODEL.DEVICE = "cuda"
    num_gpus = max(1, torch.cuda.device_count())

    # Swin-L usa più VRAM di ResNet-50
    # Con GPU da 24GB: 2 img/GPU OK
    # Con GPU da 16GB: usa 1 img/GPU e aumenta SOLVER.ACCUMULATE_GRAD_STEPS
    total_vram_gb = sum(
        torch.cuda.get_device_properties(i).total_memory
        for i in range(num_gpus)
    ) / 1e9

    if total_vram_gb >= num_gpus * 20:
        images_per_gpu = 2
    else:
        images_per_gpu = 1
        print(f"⚠️  VRAM limitata ({total_vram_gb:.0f} GB totale) → 1 img/GPU")
        print("   Se va OOM, riduci INPUT.MIN_SIZE_TRAIN a (480, 512, 576, 640)")

    cfg.SOLVER.IMS_PER_BATCH = num_gpus * images_per_gpu

    # ── Learning rate ──────────────────────────────────────────────────────────
    # Swin-L richiede LR più basso di ResNet-50 per stabilità
    # Base: 1e-4 per batch 16, scala linearmente
    base_lr = 0.0001 * (cfg.SOLVER.IMS_PER_BATCH / 16)
    cfg.SOLVER.BASE_LR = base_lr

    cfg.SOLVER.OPTIMIZER           = "ADAMW"
    cfg.SOLVER.WEIGHT_DECAY        = 0.05
    # Backbone Swin-L: LR ancora più basso (0.05x del decoder)
    # Il backbone è già pre-addestrato bene, non serve che cambi troppo
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.05

    # ── Schedule ───────────────────────────────────────────────────────────────
    # Swin-L converge più velocemente grazie al pre-training COCO
    # 10000 iter sono sufficienti per fine-tuning
    cfg.SOLVER.MAX_ITER      = 10000
    cfg.SOLVER.STEPS         = (7000, 9000)
    cfg.SOLVER.GAMMA         = 0.1
    cfg.SOLVER.WARMUP_ITERS  = 1000   # warmup breve: pesi già buoni da COCO

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED    = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE  = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

    # ── Valutazione ────────────────────────────────────────────────────────────
    cfg.TEST.EVAL_PERIOD         = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    # ── Input ──────────────────────────────────────────────────────────────────
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 576, 640, 704, 768, 832, 900)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST  = 800
    cfg.INPUT.MAX_SIZE_TEST  = 1333

    # ── Output ─────────────────────────────────────────────────────────────────
    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Stampa riepilogo
    print("\n" + "=" * 60)
    print("⚡  CONFIGURAZIONE TRAINING SWIN-L")
    print("=" * 60)
    print(f"  Backbone          : Swin-L (197M params) ← era ResNet-50 (25M)")
    print(f"  Pre-training      : COCO instance segmentation")
    print(f"  GPU disponibili   : {num_gpus}")
    print(f"  VRAM totale       : {total_vram_gb:.0f} GB")
    print(f"  Immagini/GPU      : {images_per_gpu}")
    print(f"  Batch size totale : {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Learning rate     : {cfg.SOLVER.BASE_LR:.6f}")
    print(f"  Backbone LR mult  : {cfg.SOLVER.BACKBONE_MULTIPLIER} (più conservativo)")
    print(f"  MAX_ITER          : {cfg.SOLVER.MAX_ITER}")
    print(f"  Num classi        : {num_classes}")
    print(f"  Output dir        : {cfg.OUTPUT_DIR}")
    print("=" * 60 + "\n")

    return cfg


# ==============================================================================
# 7.  MAIN
# ==============================================================================

def main(args=None):
    # Download pesi Swin-L se necessario
    download_swinl_weights()

    # Pulizia dataset
    print("\n🧹 Verifica dataset...")
    clean_and_verify_dataset(TRAIN_JSON)
    clean_and_verify_dataset(VALID_JSON)

    # Registra datasets
    for name, json_f, img_f in [
        (TRAIN_NAME, TRAIN_JSON, TRAIN_IMG),
        (VALID_NAME, VALID_JSON, VALID_IMG),
    ]:
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        register_coco_instances(name, {}, json_f, img_f)
    print(f"✅ Dataset registrati: {TRAIN_NAME}, {VALID_NAME}")

    cfg = setup()

    # Salva config
    with open(os.path.join(OUTPUT_DIR, "config_swinl.yaml"), "w") as f:
        f.write(cfg.dump())

    trainer = SolarTrainer(cfg)

    trainer.register_hooks([CSVLogHook(period=100)])

    trainer.register_hooks([
        hooks.BestCheckpointer(
            cfg.TEST.EVAL_PERIOD,
            trainer.checkpointer,
            val_metric="segm/AP",
            mode="max",
        )
    ])

    trainer.resume_or_load(resume=True)

    print("\n🚀 Avvio training Swin-L...\n")
    return trainer.train()


# ==============================================================================
# 8.  ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    num_gpus = max(1, torch.cuda.device_count())
    print(f"\n🖥️  GPU trovate: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name} — {props.total_memory / 1e9:.1f} GB VRAM")

    launch(
        main,
        num_gpus_per_machine=num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(None,),
    )
