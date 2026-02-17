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
import random  # <--- AGGIUNTO PER LA SELEZIONE CASUALE
from pyproj import Transformer

# --- CONFIGURAZIONE SISTEMA ---
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

# --- IMPORTS DETECTRON2 & MASKDINO ---
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.layers import nms

try:
    from maskdino import add_maskdino_config
except ImportError:
    print("❌ Errore: Libreria 'maskdino' non trovata. Esegui lo script dalla root del progetto.")
    sys.exit(1)

# ==============================================================================
# ⚙️ CONFIGURAZIONE PARAMETRI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_solar_16")
DATASET_FOLDER_NAME = "solar_datasets"

# Cartelle Input
VAL_IMGS = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "test") 
ORIGINAL_MOSAIC_PATH = os.path.join(BASE_DIR, "ortomosaico.tif")

# Cartelle Output
VIS_OUTPUT_DIR = os.path.join(BASE_DIR, "inference_results")
OUTPUT_MOSAIC_MARKED = "Mosaico_Finale_Rilevato.jpg"
OUTPUT_KML = "Mappa_Pannelli.kml"

# Parametri Modello
YAML_CONFIG = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation", "maskdino_R50_bs16_50ep_3s.yaml")
NUM_CLASSES = 1

# Soglie di Rilevamento
SOGLIA_DETECTION = 0.35
NMS_THRESH = 0.80
MERGE_DIST_METERS = 1

# Grafica
DOT_RADIUS_MOSAIC = 8

# ==============================================================================
# 🛠️ FUNZIONI DI SUPPORTO
# ==============================================================================

def get_best_weights(output_dir):
    best_model = os.path.join(output_dir, "model_best.pth")
    final_model = os.path.join(output_dir, "model_final.pth")
    if os.path.exists(best_model):
        print(f"✅ Modello: {os.path.basename(best_model)}")
        return best_model
    elif os.path.exists(final_model):
        print(f"⚠️ Modello: {os.path.basename(final_model)}")
        return final_model
    else:
        checkpoints = glob.glob(os.path.join(output_dir, "model_*.pth"))
        if checkpoints:
            latest = max(checkpoints, key=os.path.getctime)
            return latest
        else:
            print(f"❌ NESSUN modello .pth in {output_dir}")
            sys.exit(1)

def get_geo_tools(tif_path):
    if not os.path.exists(tif_path): return None, None
    try:
        with rasterio.open(tif_path) as src:
            affine, crs = src.transform, src.crs
        return affine, Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    except Exception as e:
        print(f"⚠️ Errore GeoTIFF: {e}")
        return None, None

def filtra_punti_unici(detections, soglia_metri):
    unique_panels = []
    detections.sort(key=lambda x: x['score'], reverse=True)
    for p in detections:
        found = False
        for up in unique_panels:
            dist = math.sqrt((p['lat'] - up['lat'])**2 + (p['lon'] - up['lon'])**2) * 111000
            if dist < soglia_metri:
                n = up['count']
                up['lat'] = (up['lat'] * n + p['lat']) / (n + 1)
                up['lon'] = (up['lon'] * n + p['lon']) / (n + 1)
                up['gx'] = (up['gx'] * n + p['gx']) / (n + 1)
                up['gy'] = (up['gy'] * n + p['gy']) / (n + 1)
                up['count'] += 1
                found = True
                break
        if not found: unique_panels.append({**p, 'count': 1})
    return unique_panels

def setup_cfg(weights_path):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskdino_config(cfg)
    cfg.merge_from_file(YAML_CONFIG)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.MaskDINO.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SOGLIA_DETECTION
    return cfg

# ==============================================================================
# 🚀 MAIN LOOP
# ==============================================================================

def main():
    print("\n--- AVVIO INFERENZA SOLAR PANELS (RANDOM MODE) ---")
    
    # 1. Setup Modello
    weights_path = get_best_weights(OUTPUT_DIR)
    cfg = setup_cfg(weights_path)
    predictor = DefaultPredictor(cfg)
    
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    
    # 2. Caricamento File
    all_files = sorted(glob.glob(os.path.join(VAL_IMGS, "*.jpg")) + glob.glob(os.path.join(VAL_IMGS, "*.JPG")))
    if not all_files:
        print(f"❌ Nessuna immagine trovata in {VAL_IMGS}")
        sys.exit(1)

    # --- RICHIESTA INPUT UTENTE ---
    print(f"📂 Trovate {len(all_files)} immagini totali.")
    scelta = input("👉 Quante immagini vuoi analizzare casualmente? (Premi INVIO per tutte): ").strip()
    
    if scelta != "":
        try:
            limite = int(scelta)
            # Evita errori se chiedi più foto di quelle presenti
            limite = min(limite, len(all_files))
            # SELEZIONE CASUALE
            all_files = random.sample(all_files, limite)
            print(f"🎲 Ho selezionato {len(all_files)} immagini a caso dal dataset.\n")
        except ValueError:
            print("⚠️ Input non valido. Analizzo tutto il dataset.\n")
    else:
        print("🚀 Analizzo tutte le immagini.\n")

    # 3. Setup Geo
    affine, geo_transformer = get_geo_tools(ORIGINAL_MOSAIC_PATH)
    raw_detections = []
    
    # 4. Loop Inferenza
    for i, path in enumerate(all_files):
        filename = os.path.basename(path)
        
        match = re.search(r"tile_col_(\d+)_row_(\d+)", filename)
        off_x, off_y = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
            
        img = cv2.imread(path)
        if img is None: continue

        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        
        if len(instances) > 0:
            keep = nms(instances.pred_boxes.tensor, instances.scores, NMS_THRESH)
            instances = instances[keep]

        num_panels = len(instances)
        max_score = instances.scores.max().item() if num_panels > 0 else 0.0
        
        print(f"📸 [{i+1}/{len(all_files)}] {filename} | Pannelli: {num_panels} | Conf: {max_score:.1%}")

        if num_panels > 0:
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
            v.overlay_instances(masks=instances.pred_masks)
            for k in range(num_panels):
                score_pct = f"{int(instances.scores[k].item() * 100)}%"
                v.draw_text(score_pct, instances.pred_boxes.tensor[k][:2])
            
            out = v.get_output()
            cv2.imwrite(os.path.join(VIS_OUTPUT_DIR, f"res_{filename}"), out.get_image()[:, :, ::-1])

            if affine and geo_transformer:
                masks = instances.pred_masks.numpy()
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.numpy()
                
                for k in range(num_panels):
                    mask_uint8 = (masks[k].astype("uint8") * 255)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    global_contour, cx_local, cy_local = None, 0, 0
                    if len(contours) > 0:
                        c = max(contours, key=cv2.contourArea)
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cx_local, cy_local = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        global_contour = c.copy()
                        global_contour[:, :, 0] += off_x
                        global_contour[:, :, 1] += off_y
                    else:
                        cx_local, cy_local = (boxes[k][0] + boxes[k][2]) / 2, (boxes[k][1] + boxes[k][3]) / 2

                    gx, gy = off_x + cx_local, off_y + cy_local
                    proj_x, proj_y = affine * (gx, gy)
                    lon, lat = geo_transformer.transform(proj_x, proj_y)
                    
                    raw_detections.append({
                        'lat': lat, 'lon': lon, 'gx': gx, 'gy': gy,
                        'score': scores[k], 'contour': global_contour
                    })

    if not raw_detections:
        print("\n❌ Nessun pannello rilevato nelle immagini selezionate.")
        sys.exit(0)
        
    print(f"\n🔄 Fusione duplicati (Soglia: {MERGE_DIST_METERS}m)...")
    final_panels = filtra_punti_unici(raw_detections, MERGE_DIST_METERS)
    
    # 5. Generazione Output Finale
    kml = simplekml.Kml()
    for idx, p in enumerate(final_panels):
        pnt = kml.newpoint(name=f"PV_{idx+1}", coords=[(p['lon'], p['lat'])])
        pnt.description = f"Score: {p['score']:.2f}"
    kml.save(OUTPUT_KML)
    print(f"✅ KML salvato: {OUTPUT_KML}")

    if os.path.exists(ORIGINAL_MOSAIC_PATH):
        try:
            mosaico = cv2.imread(ORIGINAL_MOSAIC_PATH)
            if mosaico is not None:
                for p in final_panels:
                    if p.get('contour') is not None:
                         cv2.drawContours(mosaico, [p['contour']], -1, (0, 0, 255), 2)
                    else:
                        cv2.circle(mosaico, (int(p['gx']), int(p['gy'])), DOT_RADIUS_MOSAIC, (0, 0, 255), -1)
                cv2.imwrite(OUTPUT_MOSAIC_MARKED, mosaico)
                print(f"✅ Mosaico salvato: {OUTPUT_MOSAIC_MARKED}")
        except Exception as e:
            print(f"⚠️ Errore mosaico: {e}")

    print("\n🎯 PROCEDURA COMPLETATA.")

if __name__ == "__main__":
    main()
