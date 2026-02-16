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
from detectron2.engine import DefaultTrainer, launch, HookBase
from detectron2.utils.events import get_event_storage
from detectron2.data import DatasetMapper
from maskdino import add_maskdino_config

# ==========================================================================================
# CONFIGURAZIONE PERCORSI
# ==========================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_NAME = "solar_datasets" 
TRAIN_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train", "_annotations.coco.json")
TRAIN_IMG = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train")
VALID_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json")
VALID_IMG = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid")

TRAIN_NAME = "solar_train_final"
VALID_NAME = "solar_valid_final"
OUTPUT_DIR = os.path.join(BASE_DIR, "output_solar_15")

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
    return [cat['name'] for cat in sorted(data.get('categories', []), key=lambda x: x['id'])]

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
        super().__init__(cfg, is_train=is_train)
        
        self.tfm_gens = [
            T.ResizeShortestEdge(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333, 
                sample_style="choice"
            ),
            # MODIFICA: Ridotta la rotazione da 45 a 15 gradi per stabilità
            T.RandomRotation(angle=[-15, 15], expand=False),
            # MODIFICA: Rimosso RandomCrop poiché le immagini sono già 800x800
            # T.RandomCrop("relative_range", (0.8, 0.8)), 
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomSaturation(0.8, 1.2),
            T.RandomLighting(0.7),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        ]

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=CustomTrainMapper(cfg, is_train=True))

# ==========================================================================================
# 3. SETUP CONFIGURAZIONE MULTI-GPU
# ==========================================================================================

def setup(args=None):
    cfg = get_cfg()
    cfg.set_new_allowed(True) 
    add_maskdino_config(cfg)
    
    YAML_PATH = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation", "maskdino_R50_bs16_50ep_3s.yaml")
    cfg.merge_from_file(YAML_PATH)
    
    cfg.MODEL.MASK_ON = True
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # --- LOGICA BATCH SIZE E GPU ---
    num_gpus = torch.cuda.device_count()
   
    IMAGES_PER_GPU = 2
    
    # Calcolo Batch Size Globale (Totale)
    GLOBAL_BATCH_SIZE = num_gpus * IMAGES_PER_GPU
    cfg.SOLVER.IMS_PER_BATCH = GLOBAL_BATCH_SIZE
    
    # --- SCALING DEL LEARNING RATE (MODIFICATO) ---
    # MODIFICA: Fissato LR a un valore basso e stabile per Transformer, invece di scalarlo linearmente
    cfg.SOLVER.BASE_LR = 0.0001
    
    print(f"\n⚡ CONFIGURAZIONE GPU ⚡")
    print(f"   GPUs Trovate: {num_gpus}")
    print(f"   Immagini per GPU: {IMAGES_PER_GPU}")
    print(f"   Batch Size Totale: {GLOBAL_BATCH_SIZE}")
    print(f"   Learning Rate Fisso: {cfg.SOLVER.BASE_LR:.5f}\n")

    # --- ALTRI PARAMETRI ---
    cfg.SOLVER.WEIGHT_DECAY = 0.05 
    cfg.SOLVER.MAX_ITER = 35000 
    cfg.SOLVER.STEPS = (28000, 32000)
    cfg.SOLVER.WARMUP_ITERS = 1000
    
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    
    cfg.TEST.EVAL_PERIOD = 1000 
    
    cfg.DATASETS.TRAIN = (TRAIN_NAME,)
    cfg.DATASETS.TEST = (VALID_NAME,)
    cfg.MODEL.DEVICE = "cuda" 
    
    num_classes = len(get_classes_from_json(TRAIN_JSON))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.MaskDINO.NUM_CLASSES = num_classes
    
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000 
    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

def main(args=None):
    clean_and_verify_dataset(TRAIN_JSON)
    clean_and_verify_dataset(VALID_JSON)
    
    register_coco_instances(TRAIN_NAME, {}, TRAIN_JSON, TRAIN_IMG)
    register_coco_instances(VALID_NAME, {}, VALID_JSON, VALID_IMG)
    
    cfg = setup(args)
    
    with open(os.path.join(OUTPUT_DIR, "config_used.yaml"), "w") as f:
        f.write(cfg.dump())
        
    trainer = CustomTrainer(cfg) 
    trainer.register_hooks([CSVLogHook(period=100)]) 
    trainer.resume_or_load(resume=True)
    
    return trainer.train()

if __name__ == "__main__":
    # Rileva automaticamente le GPU visibili (passate da CUDA_VISIBLE_DEVICES)
    gpus_available = torch.cuda.device_count()
    
    print("Avvio training distribuito...")
    # Lancia n processi, uno per ogni GPU disponibile
    launch(
        main, 
        num_gpus_per_machine=gpus_available,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(None,)
    )
