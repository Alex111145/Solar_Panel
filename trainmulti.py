import os
import torch
import csv
import json
import math
import logging
import detectron2.data.transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, launch, HookBase, hooks
from detectron2.utils.events import get_event_storage
from detectron2.data import DatasetMapper
from maskdino import add_maskdino_config

# ==========================================================================================
# CONFIGURAZIONE PERCORSI
# e uguale al file singola clas docosemgentation ho solo cmabiato il path
# ==========================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_NAME = "multiclass" 
TRAIN_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train", "_annotations.coco.json")
TRAIN_IMG = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train")
VALID_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json")
VALID_IMG = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid")

TRAIN_NAME = "solar_train_final"
VALID_NAME = "solar_valid_final"
OUTPUT_DIR = os.path.join(BASE_DIR, "output_solar_multiclass")

# ==========================================================================================
# 1. UTILS
# ==========================================================================================

def clean_and_verify_dataset(json_path):
    if not os.path.exists(json_path): return
    with open(json_path, 'r') as f:
        data = json.load(f)
    initial_count = len(data['annotations'])
    
    cleaned_ann = [
        ann for ann in data['annotations'] 
        if ann.get('area', 0) > 50 
        and ann['bbox'][2] > 5 
        and ann['bbox'][3] > 5 
        and len(ann.get('segmentation', [])) > 0 
    ]
    
    if initial_count - len(cleaned_ann) > 0:
        data['annotations'] = cleaned_ann
        with open(json_path, 'w') as f: 
            json.dump(data, f)
        print(f"⚠️  Pulizia {os.path.basename(json_path)}: Rimosse {initial_count - len(cleaned_ann)} annotazioni sporche.")

def get_classes_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Estrazione e stampa di ID e Nomi Classi
    categories = sorted(data.get('categories', []), key=lambda x: x['id'])
    print(f"\n📂 ANALISI CLASSI NEL JSON: {os.path.basename(json_path)}")
    print("-" * 30)
    for cat in categories:
        print(f"   [ID: {cat['id']}] -> Nome: {cat['name']}")
    print("-" * 30 + "\n")
    
    return [cat['name'] for cat in categories]

class CSVLogHook(HookBase):
    def __init__(self, period=100):
        self._period = period

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self._period == 0:
            storage = get_event_storage()
            output_file = os.path.join(self.trainer.cfg.OUTPUT_DIR, "metrics_all.csv")
            
            all_metrics = {}
            for key, history in storage.histories().items():
                try:
                    val = history.latest()
                    all_metrics[key] = 0.0 if math.isnan(val) or math.isinf(val) else val
                except:
                    continue
            
            all_metrics["iteration"] = next_iter
            
            file_exists = os.path.isfile(output_file)
            fieldnames = sorted(all_metrics.keys())
            
            with open(output_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists: writer.writeheader()
                writer.writerow(all_metrics)
            
            if next_iter % 500 == 0:
                loss = all_metrics.get('total_loss', 0)
                lr = all_metrics.get('lr', 0)
                print(f"   --> [Iter {next_iter}] Loss: {loss:.4f} | LR: {lr:.6f}")

# ==========================================================================================
# 2. MAPPER & TRAINER
# ==========================================================================================

class CustomTrainMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        # Configurato per caricare maschere di istanza (poligoni Roboflow)
        super().__init__(cfg, is_train=is_train, augmentations=[
            T.ResizeShortestEdge(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333, 
                sample_style="choice"
            ),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomRotation(angle=[-15, 15], expand=False),
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
        ],
        use_instance_mask=True,             # Fondamentale per la segmentazione
        instance_mask_format=cfg.INPUT.MASK_FORMAT)

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=CustomTrainMapper(cfg, is_train=True))

# ==========================================================================================
# 3. SETUP CONFIGURAZIONE
# ==========================================================================================

def setup(args=None):
    cfg = get_cfg()
    cfg.set_new_allowed(True) 
    add_maskdino_config(cfg)
    
    YAML_PATH = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation", "maskdino_R50_bs16_50ep_3s.yaml")
    cfg.merge_from_file(YAML_PATH)
    
    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"

    # --- SETTAGGI SEGMENTAZIONE ---
    cfg.MODEL.MASK_ON = True
    cfg.INPUT.MASK_FORMAT = "bitmask" 
    
    # --- LOGICA BATCH SIZE E GPU ---
    num_gpus = torch.cuda.device_count()
    IMAGES_PER_GPU = 2
    GLOBAL_BATCH_SIZE = max(1, num_gpus) * IMAGES_PER_GPU
    cfg.SOLVER.IMS_PER_BATCH = GLOBAL_BATCH_SIZE
    
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1 
    
    print(f"⚡ CONFIGURAZIONE GPU ⚡")
    print(f"   GPUs Trovate: {num_gpus}")
    print(f"   Batch Size Totale: {GLOBAL_BATCH_SIZE}")

    # --- PARAMETRI TRAINING ---
    cfg.SOLVER.WEIGHT_DECAY = 0.05 
    cfg.SOLVER.MAX_ITER = 8000   
    cfg.SOLVER.STEPS = (6000, 7500) 
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    
    cfg.TEST.EVAL_PERIOD = 1000 
    cfg.DATASETS.TRAIN = (TRAIN_NAME,)
    cfg.DATASETS.TEST = (VALID_NAME,)
    cfg.MODEL.DEVICE = "cuda" 
    
    # --- CONFIGURAZIONE CLASSI ---
    classes = get_classes_from_json(TRAIN_JSON)
    num_classes = len(classes)
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.MaskDINO.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.MODEL.MaskDINO.LOAD_POLY_AS_MASK = True

    cfg.SOLVER.CHECKPOINT_PERIOD = 1000 
    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

def main(args=None):
    clean_and_verify_dataset(TRAIN_JSON)
    clean_and_verify_dataset(VALID_JSON)
    
    classes = get_classes_from_json(TRAIN_JSON)
    register_coco_instances(TRAIN_NAME, {}, TRAIN_JSON, TRAIN_IMG)
    register_coco_instances(VALID_NAME, {}, VALID_JSON, VALID_IMG)
    
    MetadataCatalog.get(TRAIN_NAME).set(thing_classes=classes)
    MetadataCatalog.get(VALID_NAME).set(thing_classes=classes)
    
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
    
    trainer.resume_or_load(resume=True)
    return trainer.train()

if __name__ == "__main__":
    gpus_available = torch.cuda.device_count()
    print("Avvio training distribuito...")
    launch(
        main, 
        num_gpus_per_machine=gpus_available,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(None,)
    )
