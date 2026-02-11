#!/usr/bin/env python
import os
import cv2
import random
import torch
import warnings
import json

warnings.filterwarnings("ignore")

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from maskdino import add_maskdino_config
from detectron2.layers import nms

# ==============================================================================
# --- CONFIGURAZIONE ---
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_NAME = "solar_datasets" 

VAL_JSON   = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json")
VAL_IMGS   = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_solar_fixed")

# Cerca prima il modello finale, altrimenti l'ultimo checkpoint
if os.path.exists(os.path.join(OUTPUT_DIR, "model_final.pth")):
    WEIGHTS_FILE = os.path.join(OUTPUT_DIR, "model_final.pth")
else:
    # Se non trovi model_final, metti qui il nome esatto del tuo file .pth (es. model_0005999.pth)
    WEIGHTS_FILE = os.path.join(OUTPUT_DIR, "model_0005999.pth")

# --- PARAMETRI DI PULIZIA (AGGIORNATI) ---
# Abbassiamo drasticamente per vedere cosa "pensa" il modello
VIS_SCORE_THRESH = 0.15  # Mostra oggetti anche se la sicurezza è solo del 15%
NMS_THRESH = 0.50        # Meno aggressivo sulle sovrapposizioni

NUM_SAMPLES = 15          

def get_classes_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    categories = sorted(data.get('categories', []), key=lambda x: x['id'])
    return [cat['name'] for cat in categories]

def setup_cfg(num_classes):
    cfg = get_cfg()
    cfg.set_new_allowed(True) 
    add_maskdino_config(cfg)
    
    YAML_PATH = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation", "maskdino_R50_bs16_50ep_3s.yaml")
    cfg.merge_from_file(YAML_PATH)
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.MaskDINO.NUM_CLASSES = num_classes
    
    cfg.MODEL.WEIGHTS = WEIGHTS_FILE
    # Impostiamo a 0 nel config per non far filtrare nulla a Detectron2, filtriamo noi dopo
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.00 
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    cfg.MODEL.MaskDINO.DN = "no"
    cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
    cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
    
    cfg.freeze()
    return cfg

def main():
    print(f"\n🚀 AVVIO VISUALIZZAZIONE (Soglia bassa: {VIS_SCORE_THRESH})")
    print(f"--- Checkpoint: {os.path.basename(WEIGHTS_FILE)}")

    CLASS_NAMES = get_classes_from_json(VAL_JSON)
    num_classes = len(CLASS_NAMES)
    
    DATASET_NAME = "solar_val_inference"
    if DATASET_NAME in DatasetCatalog.list():
        DatasetCatalog.remove(DATASET_NAME)
    register_coco_instances(DATASET_NAME, {}, VAL_JSON, VAL_IMGS)
    
    metadata = MetadataCatalog.get(DATASET_NAME)
    metadata.thing_classes = CLASS_NAMES

    if not os.path.exists(WEIGHTS_FILE):
        print(f"❌ ERRORE: Pesi non trovati in {WEIGHTS_FILE}")
        return

    cfg = setup_cfg(num_classes)
    predictor = DefaultPredictor(cfg)
    
    vis_output = os.path.join(BASE_DIR, "risultati_visivi")
    os.makedirs(vis_output, exist_ok=True)
    
    dataset_dicts = DatasetCatalog.get(DATASET_NAME)
    samples = random.sample(dataset_dicts, min(NUM_SAMPLES, len(dataset_dicts)))

    print(f"--- Analisi di {len(samples)} immagini ---\n")

    for i, d in enumerate(samples):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        
        instances = outputs["instances"].to("cpu")
        
        # --- DEBUG: VEDIAMO I PUNTEGGI REALI ---
        if len(instances) > 0:
            max_score = instances.scores.max().item()
            print(f"   > Max Score rilevato dal modello: {max_score:.4f}")
        else:
            print(f"   > Nessuna istanza rilevata (strano per MaskDINO)")

        # 1. Filtro Score
        keep_idxs = instances.scores > VIS_SCORE_THRESH
        instances = instances[keep_idxs]
        
        # 2. Filtro NMS (solo se rimangono box)
        if len(instances) > 0:
            keep = nms(instances.pred_boxes.tensor, instances.scores, NMS_THRESH)
            instances = instances[keep]

        v = Visualizer(img[:, :, ::-1], 
                       metadata=metadata, 
                       scale=0.8, 
                       instance_mode=ColorMode.IMAGE)
        
        out = v.draw_instance_predictions(instances)
        
        res_name = f"pred_{os.path.basename(d['file_name'])}"
        save_path = os.path.join(vis_output, res_name)
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
        
        print(f"[{i+1}/{NUM_SAMPLES}] {res_name}: {len(instances)} oggetti salvati.")

    print(f"\n✅ Completato! Controlla i messaggi 'Max Score' qui sopra.")

if __name__ == "__main__":
    main()