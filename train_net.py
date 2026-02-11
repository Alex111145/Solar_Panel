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

TRAIN_NAME = "solar_train_v2"
VALID_NAME = "solar_valid_v2"

# --- 1. FUNZIONE DI PULIZIA DATASET ---
def clean_and_verify_dataset(json_path):
    if not os.path.exists(json_path): return
    with open(json_path, 'r') as f:
        data = json.load(f)
    initial_count = len(data['annotations'])
    cleaned_ann = [ann for ann in data['annotations'] if ann.get('area', 0) > 0.1 and ann['bbox'][2] > 0 and ann['bbox'][3] > 0]
    if initial_count - len(cleaned_ann) > 0:
        data['annotations'] = cleaned_ann
        with open(json_path, 'w') as f: json.dump(data, f)
        print(f"⚠️ Pulizia {os.path.basename(json_path)}: Rimosse annotazioni non valide.")

# --- 2. HOOK PER LOGGING DI TUTTE LE METRICHE ---
class CSVLogHook(HookBase):
    def __init__(self, period=100): # Salvataggio ogni 100 iterazioni per controllo rapido
        self._period = period

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self._period == 0:
            storage = get_event_storage()
            output_file = os.path.join(self.trainer.cfg.OUTPUT_DIR, "metrics_all.csv")
            
            # storage.latest() contiene TUTTE le metriche registrate nell'ultimo step
            all_metrics = {}
            for key, history in storage.histories().items():
                try:
                    # Prendiamo il valore più recente per ogni singola metrica esistente
                    val = history.latest()
                    # Protezione contro i NaN per non rompere il CSV
                    all_metrics[key] = 0.0 if math.isnan(val) or math.isinf(val) else val
                except:
                    continue
            
            all_metrics["iteration"] = next_iter
            
            file_exists = os.path.isfile(output_file)
            # Ordiniamo le chiavi alfabeticamente per avere un CSV ordinato
            fieldnames = sorted(all_metrics.keys())
            
            with open(output_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(all_metrics)
            print(f"✅ Tutte le metriche ({len(all_metrics)}) salvate in metrics_all.csv")

# --- 3. SETUP E TRAINING ---
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
    
    # --- CONFIGURAZIONE ANTI-CRASH ---
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True 
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value" 
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.01 
    
    cfg.DATASETS.TRAIN = (TRAIN_NAME,)
    cfg.DATASETS.TEST = (VALID_NAME,)
    cfg.MODEL.DEVICE = "cuda" 
    
    num_classes = len(get_classes_from_json(TRAIN_JSON))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.MaskDINO.NUM_CLASSES = num_classes
    
    # --- BATCH SIZE ---
    # Nota: IMS_PER_BATCH è il totale su tutte le GPU. 
    # Se usi 2 GPU e metti 4, avrai 2 immagini per GPU.
    cfg.SOLVER.IMS_PER_BATCH = 4
    
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.WARMUP_ITERS = 1000 
    cfg.SOLVER.MAX_ITER = 50000    
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000 
    
    cfg.OUTPUT_DIR = os.path.join(BASE_DIR, "output_solar_fixed")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def main(args=None):
    clean_and_verify_dataset(TRAIN_JSON)
    clean_and_verify_dataset(VALID_JSON)
    register_coco_instances(TRAIN_NAME, {}, TRAIN_JSON, TRAIN_IMG)
    register_coco_instances(VALID_NAME, {}, VALID_JSON, VALID_IMG)
    
    cfg = setup(args)
    trainer = DefaultTrainer(cfg) 
    # Registra l'Hook per salvare TUTTE le metriche
    trainer.register_hooks([CSVLogHook(period=100)]) 
    
    # --- RESUME ---
    # Se trova un checkpoint in output_solar_fixed, riparte da lì.
    trainer.resume_or_load(resume=True)
    
    return trainer.train()

if __name__ == "__main__":
    # Rilevamento iniziale delle GPU disponibili nel sistema
    n_gpus_sys = torch.cuda.device_count()
    print("\n" + "="*40)
    print(f"[ SELEZIONE GPU ] - Rilevate {n_gpus_sys} GPU nel sistema")
    for i in range(n_gpus_sys): 
        print(f"ID [{i}] : {torch.cuda.get_device_name(i)}")
    print("="*40)

    # Input utente
    choice = input("\nInserisci gli ID delle GPU da usare separati da virgola (es. '2,3' oppure '0,1'): ").strip()
    
    if not choice:
        # Se l'utente preme invio senza scrivere nulla, usa tutte le GPU
        print(f"Nessuna selezione: utilizzo tutte le {n_gpus_sys} GPU disponibili.")
        num_gpus_selected = n_gpus_sys
    else:
        # Imposta la variabile d'ambiente per rendere visibili solo quelle scelte
        os.environ["CUDA_VISIBLE_DEVICES"] = choice
        # Calcola quante GPU sono state selezionate contando gli elementi separati da virgola
        num_gpus_selected = len(choice.split(','))
        print(f"--> Utilizzo {num_gpus_selected} GPU (IDs: {choice})")

    print("\nAvvio del training distribuito...")
    
    # Launch gestisce automaticamente i processi multipli
    launch(
        main,
        num_gpus_per_machine=num_gpus_selected,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(),
    )