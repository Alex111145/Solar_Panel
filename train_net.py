import os
import torch
import csv
import json
import math
import logging
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, DatasetMapper
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, launch, HookBase
from detectron2.utils.events import get_event_storage
from detectron2.evaluation import COCOEvaluator
import detectron2.utils.comm as comm
from maskdino import add_maskdino_config

# ==============================================================================
# --- 1. CONFIGURAZIONE GENERALE ---
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_NAME = "solar_datasets" 
TRAIN_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train", "_annotations.coco.json")
TRAIN_IMG = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train")
VALID_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json")
VALID_IMG = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid")

TRAIN_NAME = "solar_train_v2"
VALID_NAME = "solar_valid_v2"

TOTAL_BATCH_SIZE = 8
BASE_LEARNING_RATE = 0.0002
MAX_ITERATIONS = 40000
WARMUP_ITERATIONS = 1000
CHECKPOINT_PERIOD = 1000
EVAL_PERIOD = 1000
OUTPUT_DIR_NAME = "output_solar_fixed_v3"

CONFIG_FILE_PATH = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation", "maskdino_R50_bs16_50ep_3s.yaml")
CLIP_GRAD_VALUE = 0.1 

# ==============================================================================

# --- 2. FUNZIONI DI UTILITÀ ---
def clean_and_verify_dataset(json_path):
    if not os.path.exists(json_path): return
    with open(json_path, 'r') as f:
        data = json.load(f)
    initial_count = len(data['annotations'])
    cleaned_ann = [ann for ann in data['annotations'] if ann.get('area', 0) > 0.1 and ann['bbox'][2] > 0 and ann['bbox'][3] > 0]
    
    if initial_count - len(cleaned_ann) > 0:
        data['annotations'] = cleaned_ann
        with open(json_path, 'w') as f: json.dump(data, f)
        print(f"⚠️ Pulizia {os.path.basename(json_path)}: Rimosse {initial_count - len(cleaned_ann)} annotazioni non valide.")

def get_classes_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [cat['name'] for cat in sorted(data.get('categories', []), key=lambda x: x['id'])]

# --- 3. HOOK PER VALIDATION LOSS ---
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        total = len(self._data_loader)
        num_batches = total
        accum_loss = 0
        
        for idx, inputs in enumerate(self._data_loader):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            with torch.no_grad():
                if isinstance(self._model, torch.nn.parallel.DistributedDataParallel):
                    metrics_dict = self._model.module(inputs)
                else:
                    metrics_dict = self._model(inputs)
                    
                losses = sum(loss for loss in metrics_dict.values())
                accum_loss += losses.item()
            
        comm.synchronize()
        # Riduzione corretta tra GPU
        all_losses = comm.all_gather(accum_loss)
        all_counts = comm.all_gather(num_batches)
        
        mean_loss = sum(all_losses) / sum(all_counts)
        
        storage = get_event_storage()
        storage.put_scalar("validation_loss", mean_loss)
        
        if comm.is_main_process():
            print(f"\n >>> Validation Loss @ Iter {self.trainer.iter}: {mean_loss:.5f} <<<\n")

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()

# --- 4. HOOK PER SALVATAGGIO CSV (FIXED) ---
class CSVLogHook(HookBase):
    def __init__(self, period=100):
        self._period = period

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self._period == 0:
            storage = get_event_storage()
            output_file = os.path.join(self.trainer.cfg.OUTPUT_DIR, "metrics_all.csv")
            
            all_metrics = {}
            # --- FIX: Uso di un blocco try-except per evitare l'errore .count ---
            for key, history in storage.histories().items():
                try:
                    val = history.latest()
                    all_metrics[key] = 0.0 if math.isnan(val) or math.isinf(val) else val
                except (AttributeError, ValueError, KeyError):
                    continue
            
            all_metrics["iteration"] = next_iter
            
            if comm.is_main_process():
                file_exists = os.path.isfile(output_file)
                keys = sorted([k for k in all_metrics.keys() if k != "iteration"])
                fieldnames = ["iteration"] + keys
                
                with open(output_file, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(all_metrics)
                print(f"[CSV] Metriche salvate a step {next_iter}.")

# --- 5. TRAINER PERSONALIZZATO ---
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        val_loader = build_detection_test_loader(
            self.cfg, 
            self.cfg.DATASETS.TEST[0],
            mapper=DatasetMapper(self.cfg, is_train=True, augmentations=[]) 
        )
        hooks.insert(-1, LossEvalHook(
            eval_period=self.cfg.TEST.EVAL_PERIOD,
            model=self.model,
            data_loader=val_loader
        ))
        hooks.insert(-1, CSVLogHook(period=100))
        return hooks

# --- 6. SETUP CONFIGURAZIONE ---
def setup(args=None):
    cfg = get_cfg()
    cfg.set_new_allowed(True) 
    add_maskdino_config(cfg)
    cfg.merge_from_file(CONFIG_FILE_PATH)
    
    cfg.MODEL.MASK_ON = True
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # Stabilizzazione Gradienti
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True 
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"  
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = CLIP_GRAD_VALUE 
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    
    # Dataset
    cfg.DATASETS.TRAIN = (TRAIN_NAME,)
    cfg.DATASETS.TEST = (VALID_NAME,)
    cfg.MODEL.DEVICE = "cuda" 
    
    classes = get_classes_from_json(TRAIN_JSON)
    num_classes = len(classes)
    print(f"Detected {num_classes} classes: {classes}")

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.MaskDINO.NUM_CLASSES = num_classes
    
    cfg.SOLVER.IMS_PER_BATCH = TOTAL_BATCH_SIZE
    cfg.SOLVER.BASE_LR = BASE_LEARNING_RATE
    cfg.SOLVER.WARMUP_ITERS = WARMUP_ITERATIONS
    cfg.SOLVER.MAX_ITER = MAX_ITERATIONS
    cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
    cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
    
    cfg.SOLVER.SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.MAX_TO_KEEP = 10 
    
    cfg.OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_DIR_NAME)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def main(args=None):
    clean_and_verify_dataset(TRAIN_JSON)
    clean_and_verify_dataset(VALID_JSON)
    
    # Reset per evitare errori se lanciato più volte nello stesso kernel
    DatasetCatalog.clear()
    MetadataCatalog.clear()
    
    register_coco_instances(TRAIN_NAME, {}, TRAIN_JSON, TRAIN_IMG)
    register_coco_instances(VALID_NAME, {}, VALID_JSON, VALID_IMG)
    
    cfg = setup(args)
    trainer = MyTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    return trainer.train()

if __name__ == "__main__":
    n_gpus_sys = torch.cuda.device_count()
    print("\n" + "="*40)
    print(f"[ SELEZIONE GPU ] - Rilevate {n_gpus_sys} GPU")
    for i in range(n_gpus_sys): 
        print(f"ID [{i}] : {torch.cuda.get_device_name(i)}")
    print("="*40)

    choice = input("\nInserisci gli ID delle GPU (es. '0,1') o premi Invio per tutte: ").strip()
    
    if choice:
        os.environ["CUDA_VISIBLE_DEVICES"] = choice
        num_gpus_selected = len(choice.split(','))
    else:
        num_gpus_selected = n_gpus_sys

    launch(
        main,
        num_gpus_per_machine=num_gpus_selected,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(),
    )
