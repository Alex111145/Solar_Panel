#!/usr/bin/env python
import os
import cv2
import torch
import re
import glob
import math
import simplekml
import json
import warnings
import numpy as np
import random

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
# ⚙️ CONFIGURAZIONE PERCORSI E PARAMETRI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATCHES_DIR = os.path.join(BASE_DIR, "my_thesis_data")
DATASET_FOLDER_NAME = "solar_datasets" 
OUTPUT_DIR = os.path.join(BASE_DIR, "output_solar_multi_gpu")

# Configurazione Dataset
VAL_JSON   = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json")
VAL_IMGS   = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid")
YAML_CONFIG = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation", "maskdino_R50_bs16_50ep_3s.yaml")

# --- RICERCA AUTOMATICA PESI ---
if os.path.exists(os.path.join(OUTPUT_DIR, "model_final.pth")):
    WEIGHTS_FILE = os.path.join(OUTPUT_DIR, "model_final.pth")
else:
    checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "model_*.pth"))
    if checkpoints:
        WEIGHTS_FILE = max(checkpoints, key=os.path.getctime)
    else:
        # Fallback basato sulla ricerca manuale presente nel codice originale
        WEIGHTS_FILE = os.path.join(OUTPUT_DIR, "model_0007999.pth")

# File Geografici
WORLD_FILE_PATH = os.path.join(BASE_DIR, "ortomosaico.tfw")
ORIGINAL_MOSAIC_PATH = os.path.join(BASE_DIR, "ortomosaico.tif")

# Output
OUTPUT_MOSAIC_MARKED = "Mosaico_Punti_Rilevati.jpg"

# DATI ANCORA
ANCHOR_PIXEL_X = 3255.0  
ANCHOR_PIXEL_Y = 1614.0  
ANCHOR_GPS_LAT = 45.840750
ANCHOR_GPS_LON = 8.790400

# Parametri Inferenza e Pulizia
NUM_CLASSES = 1
CONFIDENCE_THRESH = 0.50  
MERGE_DIST_METERS = 1.0
VIS_SCORE_THRESH = 0.15 
DOT_RADIUS_MOSAIC = 8
NMS_THRESH = 0.50        
NUM_SAMPLES = 15          

# ==============================================================================
# 🛠️ FUNZIONI DI SUPPORTO
# ==============================================================================

def get_tfw_params():
    if not os.path.exists(WORLD_FILE_PATH):
        print(f"❌ ERRORE: Manca il file {WORLD_FILE_PATH}")
        exit()
    with open(WORLD_FILE_PATH, 'r') as f:
        return [float(l.strip()) for l in f.readlines()]

def calculate_correction(tfw):
    A, D, B, E, C, F = tfw
    theo_lon = C + (ANCHOR_PIXEL_X * A) + (ANCHOR_PIXEL_Y * B)
    theo_lat = F + (ANCHOR_PIXEL_X * D) + (ANCHOR_PIXEL_Y * E)
    return (ANCHOR_GPS_LAT - theo_lat, ANCHOR_GPS_LON - theo_lon)

def get_classes_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    categories = sorted(data.get('categories', []), key=lambda x: x['id'])
    return [cat['name'] for cat in categories]

# ==============================================================================
# 🧠 CONFIGURAZIONE MODELLO (SETUP CFG)
# ==============================================================================

def setup_predictor():
    """Versione specifica per main()"""
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskdino_config(cfg)
    cfg.merge_from_file(YAML_CONFIG)
    cfg.MODEL.WEIGHTS = WEIGHTS_FILE
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESH
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.MaskDINO.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = NUM_CLASSES
    return DefaultPredictor(cfg)

def setup_cfg(num_classes):
    """Versione specifica per main2()"""
    cfg = get_cfg()
    cfg.set_new_allowed(True) 
    add_maskdino_config(cfg)
    
    cfg.merge_from_file(YAML_CONFIG)
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.MaskDINO.NUM_CLASSES = num_classes
    
    cfg.MODEL.WEIGHTS = WEIGHTS_FILE
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.00 
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    cfg.MODEL.MaskDINO.DN = "no"
    cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
    cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
    
    cfg.freeze()
    return cfg

# ==============================================================================
# 🚀 FUNZIONI PRINCIPALI
# ==============================================================================

def foto():
    """Visualizzazione risultati su campioni casuali del dataset"""
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
        
        if len(instances) > 0:
            max_score = instances.scores.max().item()
            print(f"   > Max Score rilevato dal modello: {max_score:.4f}")
        else:
            print(f"   > Nessuna istanza rilevata (strano per MaskDINO)")

        keep_idxs = instances.scores > VIS_SCORE_THRESH
        instances = instances[keep_idxs]
        
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

def main():
        
    foto()
    """Elaborazione patch geografiche, KML e Mosaico finale"""
    print("\n🌍 Caricamento parametri geografici e calibrazione ancora...")
    tfw = get_tfw_params()
    lat_corr, lon_corr = calculate_correction(tfw)
    print(f"✅ Calibrazione: Offset Lat {lat_corr:.8f}, Lon {lon_corr:.8f}")
   
    print(f"📂 Cerco patch in: {os.path.abspath(PATCHES_DIR)}")
    files = glob.glob(os.path.join(PATCHES_DIR, "*.jpg")) + glob.glob(os.path.join(PATCHES_DIR, "*.JPG"))
   
    if not files:
        print(f"⚠️ ATTENZIONE: Nessun file trovato in {PATCHES_DIR}. Controlla che le foto siano lì dentro!")
        return

    predictor = setup_predictor()

    raw_detections = []
    print(f"🔍 Analisi di {len(files)} patch in corso...")
   
    for i, path in enumerate(files):
        fname = os.path.basename(path)
        match = re.search(r"tile_col_(\d+)_row_(\d+)", fname)
        if not match: continue
        off_x, off_y = int(match.group(1)), int(match.group(2))

        img = cv2.imread(path)
        if img is None: continue
       
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
       
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.numpy()
            for box in boxes:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                gx, gy = off_x + cx, off_y + cy
                lon = tfw[4] + (gx * tfw[0]) + (gy * tfw[2]) + lon_corr
                lat = tfw[5] + (gx * tfw[1]) + (gy * tfw[3]) + lat_corr
                raw_detections.append({'lat': lat, 'lon': lon, 'gx': gx, 'gy': gy})
       
        if (i+1) % 100 == 0:
            print(f"   [PROCESSO] Analizzate {i+1}/{len(files)} immagini...")

    # Deduplicazione
    unique_panels = []
    for p in raw_detections:
        found = False
        for up in unique_panels:
            dist = math.sqrt((p['lat']-up['lat'])**2 + (p['lon']-up['lon'])**2) * 111000
            if dist < MERGE_DIST_METERS:
                n = up['count']
                up['lat'] = (up['lat']*n + p['lat'])/(n+1)
                up['lon'] = (up['lon']*n + p['lon'])/(n+1)
                up['gx'] = (up['gx']*n + p['gx'])/(n+1)
                up['gy'] = (up['gy']*n + p['gy'])/(n+1)
                up['count'] += 1
                found = True
                break
        if not found:
            unique_panels.append({'lat':p['lat'], 'lon':p['lon'], 'gx':p['gx'], 'gy':p['gy'], 'count':1})

    # Creazione KML
    print(f"🗺️ Creazione file KML per Google Earth...")
    kml = simplekml.Kml()
    for i, p in enumerate(unique_panels):
        kml.newpoint(name=f"P_{i+1}", coords=[(p['lon'], p['lat'])])
    kml.save("Mappa_Tesi_Finale.kml")

    # Creazione Mosaico con punti
    print(f"🖼️ Disegno punti sul mosaico (Rilevati: {len(unique_panels)})...")
    mosaico = cv2.imread(ORIGINAL_MOSAIC_PATH)
    if mosaico is not None:
        for p in unique_panels:
           cv2.circle(mosaico, (int(p['gx']), int(p['gy'])), DOT_RADIUS_MOSAIC, (0, 0, 255), -1)
        cv2.imwrite(OUTPUT_MOSAIC_MARKED, mosaico)
        print(f"✅ Mosaico salvato: {OUTPUT_MOSAIC_MARKED}")
    else:
        print(f"⚠️ ATTENZIONE: Impossibile caricare il mosaico da {ORIGINAL_MOSAIC_PATH}")
   
    print("-" * 30)
    print(f"✅ FINITO!")
    print(f"📍 KML: Mappa_Tesi_Finale.kml | Pannelli: {len(unique_panels)}")
    print(f"🖼️ Mosaico: {OUTPUT_MOSAIC_MARKED}")
    print("-" * 30)


if __name__ == "__main__":
  
    main()
