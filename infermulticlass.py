#!/usr/bin/env python
import os
import sys
import glob
import re
import math
import warnings
import cv2
import torch
import numpy as np
import rasterio
import simplekml
import random 
import json
from pyproj import Transformer

# --- CONFIGURAZIONE SISTEMA ---
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.layers import nms
from detectron2.structures import Instances, Boxes

try:
    from maskdino import add_maskdino_config
except ImportError:
    print("❌ Errore: Libreria 'maskdino' non trovata.")
    sys.exit(1)

# ==============================================================================
# ⚙️ CONFIGURAZIONE PARAMETRI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_solar_multiclass")
DATASET_FOLDER_NAME = "multiclass"

VAL_IMGS = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "test") 
TRAIN_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train", "_annotations.coco.json")

VIS_OUTPUT_DIR = os.path.join(BASE_DIR, "inference_results")
YAML_CONFIG = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation", "maskdino_R50_bs16_50ep_3s.yaml")

# --- SOGLIE ---
SOGLIA_DETECTION = 0.20  
NMS_THRESH = 0.05        
DIST_MERGE_PIXELS = 25   

# ==============================================================================
# 🛠️ FUNZIONI DI SUPPORTO
# ==============================================================================

def get_classes_from_json(json_path):
    if not os.path.exists(json_path):
        return ["Bird Dirt", "Clean", "Cracks", "Moisture", "Soiling"]
    with open(json_path, 'r') as f:
        data = json.load(f)
    categories = sorted(data.get('categories', []), key=lambda x: x['id'])
    return [cat['name'] for cat in categories]

def merge_masks_by_class(instances, img_shape, dist_kernel=25):
    """ Fonde maschere della stessa classe e crea maschere booleane piene """
    if len(instances) == 0: return instances, []

    classes = instances.pred_classes.numpy()
    masks = instances.pred_masks.numpy()
    scores = instances.scores.numpy()
    
    merged_data = []
    unique_classes = np.unique(classes)

    for cls in unique_classes:
        cls_idx = np.where(classes == cls)[0]
        combined_mask = np.zeros(img_shape[:2], dtype=np.uint8)
        
        for idx in cls_idx:
            combined_mask = cv2.bitwise_or(combined_mask, masks[idx].astype(np.uint8))

        # Unione dei frammenti
        kernel = np.ones((dist_kernel, dist_kernel), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 200: continue
            
            # Creazione maschera binaria solida (booleana)
            new_mask_u8 = np.zeros(img_shape[:2], dtype=np.uint8)
            cv2.drawContours(new_mask_u8, [cnt], -1, 1, -1)
            new_mask_bool = new_mask_u8 > 0

            # Calcolo score massimo tra i pezzi originali
            final_score = max([scores[idx] for idx in cls_idx])
            x, y, w, h = cv2.boundingRect(cnt)
            
            merged_data.append({
                'mask': new_mask_bool, 'class': cls, 'score': final_score, 'box': [x, y, x+w, y+h]
            })

    new_inst = Instances(img_shape[:2])
    if not merged_data: return new_inst, []
    
    new_inst.pred_classes = torch.tensor([d['class'] for d in merged_data])
    new_inst.scores = torch.tensor([d['score'] for d in merged_data])
    new_inst.pred_masks = torch.tensor(np.array([d['mask'] for d in merged_data]))
    new_inst.pred_boxes = Boxes(torch.tensor([d['box'] for d in merged_data]))
    return new_inst, merged_data

def setup_cfg(weights_path, num_classes):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskdino_config(cfg)
    cfg.merge_from_file(YAML_CONFIG)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.MaskDINO.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SOGLIA_DETECTION
    return cfg

# ==============================================================================
# 🚀 MAIN LOOP
# ==============================================================================

def main():
    class_names = get_classes_from_json(TRAIN_JSON)
    checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "*.pth"))
    weights = max(checkpoints, key=os.path.getctime) if checkpoints else sys.exit(1)
    
    cfg = setup_cfg(weights, len(class_names))
    predictor = DefaultPredictor(cfg)
    
    MetadataCatalog.get("solar_final_multi").set(thing_classes=class_names)
    metadata = MetadataCatalog.get("solar_final_multi")

    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    all_files = sorted(glob.glob(os.path.join(VAL_IMGS, "*.jpg")))

    print(f"\n🚀 Analisi di {len(all_files)} immagini...")

    for i, path in enumerate(all_files):
        img = cv2.imread(path)
        if img is None: continue
        filename = os.path.basename(path)

        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        
        if len(instances) > 0:
            # Filtro soglia manuale
            instances = instances[instances.scores >= SOGLIA_DETECTION]
            # Unione maschere
            instances, _ = merge_masks_by_class(instances, img.shape, DIST_MERGE_PIXELS)

            if len(instances) > 0:
                max_prob = torch.max(instances.scores).item()
                print(f"📸 [{i+1}/{len(all_files)}] {filename} -> {len(instances)} aree unite. (Prob Max: {max_prob:.4f})")
                
                # --- DISEGNO CORRETTO ---
                v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
                
                # 1. Disegna le maschere colorate
                v.overlay_instances(masks=instances.pred_masks, alpha=0.5)
                
                # 2. Scrive il testo sopra le maschere
                for k in range(len(instances)):
                    cls = class_names[instances.pred_classes[k]]
                    score = int(instances.scores[k] * 100)
                    v.draw_text(f"{cls} {score}%", instances.pred_boxes.tensor[k][:2])
                
                # 3. Estrae l'immagine finale e salva
                v_img = v.get_output()
                cv2.imwrite(os.path.join(VIS_OUTPUT_DIR, f"res_{filename}"), v_img.get_image()[:, :, ::-1])
            else:
                print(f"📸 [{i+1}/{len(all_files)}] {filename} -> Nessun oggetto sopra soglia.")

if __name__ == "__main__":
    main()
