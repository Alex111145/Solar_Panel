#!/usr/bin/env python
import os
import cv2
import torch
import re
import glob
import math
import simplekml
import warnings
import numpy as np

# --- CONFIGURAZIONE AMBIENTE ---
warnings.filterwarnings("ignore")

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from maskdino import add_maskdino_config
from detectron2.layers import nms

# ==============================================================================
# ⚙️ CONFIGURAZIONE PARAMETRI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATCHES_DIR = os.path.join(BASE_DIR, "my_thesis_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_solar_multi_gpu")
VIS_OUTPUT_DIR = os.path.join(BASE_DIR, "risultati_visivi") 

SOGLIA_DETECTION = 0.25  
NMS_THRESH = 0.40          
MERGE_DIST_METERS = 1.0   

DATASET_FOLDER_NAME = "solar_datasets"
VAL_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json")
VAL_IMGS = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid")
YAML_CONFIG = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation", "maskdino_R50_bs16_50ep_3s.yaml")

checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "model_*.pth"))
WEIGHTS_FILE = max(checkpoints, key=os.path.getctime) if checkpoints else os.path.join(OUTPUT_DIR, "model_final.pth")

WORLD_FILE_PATH = os.path.join(BASE_DIR, "ortomosaico.tfw")
ORIGINAL_MOSAIC_PATH = os.path.join(BASE_DIR, "ortomosaico.tif")
OUTPUT_MOSAIC_MARKED = "Mosaico_Punti_Rilevati.jpg"

ANCHOR_PIXEL_X, ANCHOR_PIXEL_Y = 3255.0, 1614.0
ANCHOR_GPS_LAT, ANCHOR_GPS_LON = 45.840750, 8.790400

NUM_CLASSES = 1
DOT_RADIUS_MOSAIC = 8  

# ==============================================================================
# 🛠️ FUNZIONI TECNICHE
# ==============================================================================

def get_tfw_params():
    with open(WORLD_FILE_PATH, 'r') as f:
        return [float(l.strip()) for l in f.readlines()]

def calculate_correction(tfw):
    A, D, B, E, C, F = tfw
    theo_lon = C + (ANCHOR_PIXEL_X * A) + (ANCHOR_PIXEL_Y * B)
    theo_lat = F + (ANCHOR_PIXEL_X * D) + (ANCHOR_PIXEL_Y * E)
    return (ANCHOR_GPS_LAT - theo_lat, ANCHOR_GPS_LON - theo_lon)

def get_mask_center_precise(mask, box):
    mask_uint8 = (mask.astype("uint8") * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        (center_x, center_y), _, _ = rect
        return center_x, center_y
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

def filtra_punti_unici(detections, soglia_metri):
    unique_panels = []
    for p in detections:
        found = False
        for up in unique_panels:
            dist = math.sqrt((p['lat']-up['lat'])**2 + (p['lon']-up['lon'])**2) * 111000
            if dist < soglia_metri:
                n = up['count']
                up['lat'] = (up['lat']*n + p['lat'])/(n+1)
                up['lon'] = (up['lon']*n + p['lon'])/(n+1)
                up['gx'] = (up['gx']*n + p['gx'])/(n+1)
                up['gy'] = (up['gy']*n + p['gy'])/(n+1)
                up['count'] += 1
                found = True
                break
        if not found:
            unique_panels.append({**p, 'count': 1})
    return unique_panels

def setup_cfg():
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskdino_config(cfg)
    cfg.merge_from_file(YAML_CONFIG)
    cfg.MODEL.WEIGHTS = WEIGHTS_FILE
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.MaskDINO.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SOGLIA_DETECTION
    return cfg

# ==============================================================================
# 🚀 MAIN PROCESS
# ==============================================================================

def main():
    all_files = sorted(glob.glob(os.path.join(PATCHES_DIR, "*.jpg")) + glob.glob(os.path.join(PATCHES_DIR, "*.JPG")))
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    
    print(f"\n📂 Patch totali: {len(all_files)}")
    try:
        n_choice = int(input("👉 Quante patch vuoi analizzare? (0 per tutte): "))
        files_to_process = all_files[:n_choice] if n_choice > 0 else all_files
    except:
        files_to_process = all_files

    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    if "solar_val" not in DatasetCatalog.list():
        register_coco_instances("solar_val", {}, VAL_JSON, VAL_IMGS)
    metadata = MetadataCatalog.get("solar_val")
    
    tfw = get_tfw_params()
    lat_corr, lon_corr = calculate_correction(tfw)
    raw_detections = []

    print(f"\n🔍 Analisi in corso...")
    for i, path in enumerate(files_to_process):
        fname = os.path.basename(path)
        match = re.search(r"tile_col_(\d+)_row_(\d+)", fname)
        if not match: continue
        off_x, off_y = int(match.group(1)), int(match.group(2))

        img = cv2.imread(path)
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        
        keep = instances.scores >= SOGLIA_DETECTION
        instances = instances[keep]
        
        if len(instances) > 0:
            keep_nms = nms(instances.pred_boxes.tensor, instances.scores, NMS_THRESH)
            instances = instances[keep_nms]
            
            print(f"   [{i+1}/{len(files_to_process)}] {fname} -> Max Score: {instances.scores.max().item():.4f}")

            # Preparazione dati per mosaico e KML
            masks_np = instances.pred_masks.numpy()
            boxes_np = instances.pred_boxes.tensor.numpy()

            for j in range(len(instances)):
                cx, cy = get_mask_center_precise(masks_np[j], boxes_np[j])
                gx, gy = off_x + cx, off_y + cy
                lon = tfw[4] + (gx * tfw[0]) + (gy * tfw[2]) + lon_corr
                lat = tfw[5] + (gx * tfw[1]) + (gy * tfw[3]) + lat_corr
                raw_detections.append({'lat': lat, 'lon': lon, 'gx': gx, 'gy': gy})

            # --- VISUALIZZAZIONE PATCH ---
            # Modifica label: trasforma i punteggi in percentuale (es. 0.85 -> 85%)
            scores = instances.scores.tolist()
            labels = [f"{int(s * 100)}%" for s in scores]

            v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
            
            # Disegna solo maschere e label (niente box)
            v._default_font_size = 18 
            out = v.draw_instance_predictions(instances)
            
            # Sovrascriviamo il disegno per assicurarci che i box non ci siano
            # Nota: draw_instance_predictions di default usa i settaggi del Visualizer
            # Per forzare solo maschere usiamo questo approccio:
            v = Visualizer(img[:, :, ::-1], metadata=metadata)
            out = v.overlay_instances(
                masks=instances.pred_masks,
                labels=labels,
                assigned_colors=None, 
                alpha=0.5
            )
            
            cv2.imwrite(os.path.join(VIS_OUTPUT_DIR, f"res_{fname}"), out.get_image()[:, :, ::-1])

    # 4. Output Finali
    final_panels = filtra_punti_unici(raw_detections, MERGE_DIST_METERS)
    
    kml = simplekml.Kml()
    mosaico = cv2.imread(ORIGINAL_MOSAIC_PATH)

    for idx, p in enumerate(final_panels):
        kml.newpoint(name=f"P_{idx+1}", coords=[(p['lon'], p['lat'])])
        if mosaico is not None:
            # Pallino rosso solo sul mosaico
            cv2.circle(mosaico, (int(p['gx']), int(p['gy'])), DOT_RADIUS_MOSAIC, (0, 0, 255), -1)

    kml.save("Mappa_Rilevamenti.kml")
    if mosaico is not None:
        cv2.imwrite(OUTPUT_MOSAIC_MARKED, mosaico)

    print("\n" + "="*30)
    print(f"✅ FOTO SALVATE IN: {VIS_OUTPUT_DIR}")
    print(f"✅ FILE GOOGLE EARTH: Mappa_Rilevamenti.kml")
    print(f"✅ MOSAICO: {OUTPUT_MOSAIC_MARKED}")
    print(f"🎯 PANNELLI TOTALI: {len(final_panels)}")
    print("="*30)

if __name__ == "__main__":
    main()
