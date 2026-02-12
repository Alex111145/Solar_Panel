import os
import torch
import csv
import json
import math
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, launch, HookBase
from detectron2.utils.events import get_event_storage
from maskdino import add_maskdino_config

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_NAME = "solar_datasets" 
TRAIN_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train", "_annotations.coco.json")
TRAIN_IMG = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train")
VALID_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json")
VALID_IMG = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid")

TRAIN_NAME = "solar_train_v3"
VALID_NAME = "solar_valid_v3"

# --- 1. FUNZIONE DI PULIZIA DATASET MIGLIORATA ---
def clean_and_verify_dataset(json_path):
    if not os.path.exists(json_path): return
    with open(json_path, 'r') as f:
        data = json.load(f)
    initial_count = len(data['annotations'])
    
    # Rimuovi annotazioni troppo piccole o malformate
    cleaned_ann = [
        ann for ann in data['annotations'] 
        if ann.get('area', 0) > 50  # Aumentato threshold area minima
        and ann['bbox'][2] > 5      # Larghezza minima
        and ann['bbox'][3] > 5      # Altezza minima
        and len(ann.get('segmentation', [])) > 0  # Deve avere segmentazione
    ]
    
    if initial_count - len(cleaned_ann) > 0:
        data['annotations'] = cleaned_ann
        with open(json_path, 'w') as f: 
            json.dump(data, f)
        print(f"⚠️ Pulizia {os.path.basename(json_path)}: Rimosse {initial_count - len(cleaned_ann)} annotazioni non valide.")

# --- 2. HOOK PER LOGGING DETTAGLIATO ---
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
                if not file_exists:
                    writer.writeheader()
                writer.writerow(all_metrics)
            
            # Stampa metriche chiave ogni 500 iter
            if next_iter % 500 == 0:
                total_loss = all_metrics.get('total_loss', 0)
                mask_dice = all_metrics.get('loss_dice', 0)
                print(f"\n[Iter {next_iter}] total_loss={total_loss:.4f}, mask_dice={mask_dice:.4f}")

# --- 3. SETUP MIGLIORATO ---
def get_classes_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [cat['name'] for cat in sorted(data.get('categories', []), key=lambda x: x['id'])]

def setup(args=None):
    cfg = get_cfg()
    cfg.set_new_allowed(True) 
    add_maskdino_config(cfg)
    
    YAML_PATH = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation", "maskdino_R50_bs16_50ep_3s.yaml")
    cfg.merge_from_file(YAML_PATH)
    
    cfg.MODEL.MASK_ON = True
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # ==================== MODIFICHE CRITICHE ====================
    
    # 1. GRADIENT CLIPPING più permissivo
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True 
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"  # Cambiato da "value" a "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0    # Più permissivo
    
    # 2. LEARNING RATE AUMENTATO
    cfg.SOLVER.BASE_LR = 0.00025  # Era 0.0001, raddoppiato
    cfg.SOLVER.WARMUP_ITERS = 500  # Warmup più breve
    cfg.SOLVER.MAX_ITER = 50000
    
    # 3. LR SCHEDULER con decay steps
    cfg.SOLVER.STEPS = (30000, 45000)  # Riduce LR a iter 30k e 45k
    cfg.SOLVER.GAMMA = 0.1             # Riduzione del 90%
    
    # 4. BATCH SIZE (aumenta se hai abbastanza VRAM)
    cfg.SOLVER.IMS_PER_BATCH = 2  # Prova con 2, torna a 1 se OOM
    
    # 5. DATA AUGMENTATION più aggressiva
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # 6. WEIGHT DECAY ridotto per evitare underfitting
    cfg.SOLVER.WEIGHT_DECAY = 0.0001  # Default 0.0001, ok
    
    # 7. TEST EVALUATION più frequente
    cfg.TEST.EVAL_PERIOD = 1000  # Valuta ogni 1000 iter
    
    # ============================================================
    
    cfg.DATASETS.TRAIN = (TRAIN_NAME,)
    cfg.DATASETS.TEST = (VALID_NAME,)
    cfg.MODEL.DEVICE = "cuda" 
    
    num_classes = len(get_classes_from_json(TRAIN_JSON))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.MaskDINO.NUM_CLASSES = num_classes
    
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000 
    
    cfg.OUTPUT_DIR = os.path.join(BASE_DIR, "output_solar_improved")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

def main(args=None):
    print("\n" + "="*60)
    print("🚀 TRAINING MIGLIORATO - Parametri ottimizzati")
    print("="*60 + "\n")
    
    clean_and_verify_dataset(TRAIN_JSON)
    clean_and_verify_dataset(VALID_JSON)
    
    register_coco_instances(TRAIN_NAME, {}, TRAIN_JSON, TRAIN_IMG)
    register_coco_instances(VALID_NAME, {}, VALID_JSON, VALID_IMG)
    
    cfg = setup(args)
    
    print("📊 CONFIGURAZIONE TRAINING:")
    print(f"   - Learning Rate: {cfg.SOLVER.BASE_LR}")
    print(f"   - Batch Size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"   - Max Iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"   - LR Decay Steps: {cfg.SOLVER.STEPS}")
    print(f"   - Warmup Iterations: {cfg.SOLVER.WARMUP_ITERS}")
    print(f"   - Output: {cfg.OUTPUT_DIR}\n")
    
    trainer = DefaultTrainer(cfg) 
    trainer.register_hooks([CSVLogHook(period=100)]) 
    
    # Resume da checkpoint se esiste
    trainer.resume_or_load(resume=True)
    
    print("🏋️ Avvio training...\n")
    return trainer.train()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print("\n[ SELEZIONE GPU ]")
    for i in range(n_gpus): 
        print(f"[{i}] {torch.cuda.get_device_name(i)}")
    choice = input("Scegli ID GPU: ").strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = choice
    launch(main, num_gpus_per_machine=1)
