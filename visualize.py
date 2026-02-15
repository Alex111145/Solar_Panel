#!/usr/bin/env python
import os

# Blocca i warning di OpenCV per i tag TIFF sconosciuti prima di caricare cv2
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import torch
import re
import glob
import math
import simplekml
import warnings
import numpy as np
import rasterio
import sys
from pyproj import Transformer

# --- CONFIGURAZIONE AMBIENTE ---
warnings.filterwarnings("ignore")

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from maskdino import add_maskdino_config
from detectron2.layers import nms

# ==============================================================================
# ⚙️ CONFIGURAZIONE PARAMETRI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_solar_multi_gpu")
VIS_OUTPUT_DIR = os.path.join(BASE_DIR, "risultati_visivi") 

SOGLIA_DETECTION = 0.35 
NMS_THRESH = 0.40           
MERGE_DIST_METERS = 0.8  

DATASET_FOLDER_NAME = "solar_datasets"
# CARTELLA Test: Usata come sorgente per le patch
VAL_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "tewst", "_annotations.coco.json")
VAL_IMGS = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "test")

YAML_CONFIG = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation", "maskdino_R50_bs16_50ep_3s.yaml")
checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "model_*.pth"))
WEIGHTS_FILE = max(checkpoints, key=os.path.getctime) if checkpoints else os.path.join(OUTPUT_DIR, "model_final.pth")

ORIGINAL_MOSAIC_PATH = os.path.join(BASE_DIR, "ortomosaico.tif")
OUTPUT_MOSAIC_MARKED = "Mosaico_Punti_Rilevati.jpg"

NUM_CLASSES = 1
DOT_RADIUS_MOSAIC = 6  

# ==============================================================================
# 🛠️ FUNZIONI TECNICHE
# ==============================================================================

def get_geo_tools(tif_path):
    """Estrae i metadati geografici reali dal TIF."""
    with rasterio.open(tif_path) as src:
        affine = src.transform
        crs = src.crs
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return affine, transformer

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
# 🚀 MAIN
# ==============================================================================

def main():
    # Carica solo i file dalla cartella test
    all_files = sorted(glob.glob(os.path.join(VAL_IMGS, "*.jpg")) + glob.glob(os.path.join(VAL_IMGS, "*.JPG")))
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    
    if not all_files:
        print(f"❌ Nessun file trovato in {VAL_IMGS}. Verifica il percorso.")
        sys.exit(1)

    print(f"\n📂 Analisi di {len(all_files)} patch dalla cartella test...")
    try:
        n_input = input("👉 Quante patch vuoi analizzare? (Invio per tutte): ")
        n_choice = int(n_input) if n_input.strip() else 0
        files_to_process = all_files[:n_choice] if n_choice > 0 else all_files
    except EOFError:
        files_to_process = all_files

    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    
    if "solar_test" not in DatasetCatalog.list():
        register_coco_instances("solar_test", {}, VAL_JSON, VAL_IMGS)
    metadata = MetadataCatalog.get("solar_test")
    
    # Inizializzazione Geografica
    try:
        affine, geo_transformer = get_geo_tools(ORIGINAL_MOSAIC_PATH)
    except Exception as e:
        print(f"❌ Errore caricamento TIF: {e}")
        sys.exit(1)

    raw_detections = []

    print("-" * 60)
    for i, path in enumerate(files_to_process):
        fname = os.path.basename(path)
        match = re.search(r"tile_col_(\d+)_row_(\d+)", fname)
        if not match: continue
        off_x, off_y = int(match.group(1)), int(match.group(2))

        img = cv2.imread(path)
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        
        # Filtro soglia esplicito
        instances = instances[instances.scores > SOGLIA_DETECTION]
        
        if len(instances) > 0:
            # Filtro NMS
            keep_nms = nms(instances.pred_boxes.tensor, instances.scores, NMS_THRESH)
            instances = instances[keep_nms]
            
            # Log terminale
            max_score = instances.scores.max().item()
            print(f"📸 [{i+1}/{len(files_to_process)}] {fname} | Pannelli: {len(instances)} | Accuracy Max: {max_score:.2%}")

            masks_np = instances.pred_masks.numpy()
            boxes_np = instances.pred_boxes.tensor.numpy()
            scores_list = instances.scores.tolist()

            for j in range(len(instances)):
                cx, cy = get_mask_center_precise(masks_np[j], boxes_np[j])
                gx, gy = off_x + cx, off_y + cy
                
                proj_x, proj_y = affine * (gx, gy)
                lon, lat = geo_transformer.transform(proj_x, proj_y)
                raw_detections.append({'lat': lat, 'lon': lon, 'gx': gx, 'gy': gy})

            # Visualizzazione con label accuracy
            v = Visualizer(img[:, :, ::-1], metadata=metadata)
            labels = [f"{s:.1%}" for s in scores_list]
            out = v.overlay_instances(masks=instances.pred_masks, labels=labels, alpha=0.4)
            cv2.imwrite(os.path.join(VIS_OUTPUT_DIR, f"res_{fname}"), out.get_image()[:, :, ::-1])
        else:
            print(f"📸 [{i+1}/{len(files_to_process)}] {fname} | Nessun pannello sopra soglia {SOGLIA_DETECTION}")

    # --- OUTPUT FINALI ---
    final_panels = filtra_punti_unici(raw_detections, MERGE_DIST_METERS)
    
    kml = simplekml.Kml()
    mosaico = cv2.imread(ORIGINAL_MOSAIC_PATH)

    # Stile pallini rossi Google Earth
    shared_style = simplekml.Style()
    shared_style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    shared_style.iconstyle.color = 'ff0000ff'  
    shared_style.iconstyle.scale = 0.7         
    shared_style.labelstyle.scale = 0          

    for idx, p in enumerate(final_panels):
        pnt = kml.newpoint(name=f"P_{idx+1}", coords=[(p['lon'], p['lat'])])
        pnt.style = shared_style
        
        if mosaico is not None:
            cv2.circle(mosaico, (int(p['gx']), int(p['gy'])), DOT_RADIUS_MOSAIC, (0, 0, 255), -1)

    kml.save("Mappa_Rilevamenti.kml")
    if mosaico is not None:
        cv2.imwrite(OUTPUT_MOSAIC_MARKED, mosaico)

    print("\n" + "="*50)
    print(f"🎯 ANALISI COMPLETATA")
    print(f"🎯 PANNELLI UNICI TROVATI: {len(final_panels)}")
    print(f"🌍 KML GENERATO: Mappa_Rilevamenti.kml")
    print("="*50)
    
    # Uscita definitiva
    sys.exit(0)

if __name__ == "__main__":
    main()
