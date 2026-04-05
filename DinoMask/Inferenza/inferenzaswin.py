#!/usr/bin/env python
"""
inference.py - Inferenza MaskDINO Solar Panels (v4 - FINAL)
============================================================
Classi aggiornate: "Pannello Sano" e "Pannello Danneggiato".
Coerente con il training v4:
  - Risoluzione test 1600px (identica al training)
  - Soglia detection 0.37
  - NMS post-processing
  - Geolocalizzazione GeoTIFF → CSV
  - Fusione duplicati per distanza
  - Calcolo area degradata (GSD²)
  - Selezione casuale N immagini
  - Mosaico annotato con contorni colorati per classe
  - API PVGIS e Meteo automatiche tramite centroide
"""

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
import random
import pandas as pd
from pyproj import Transformer

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.layers import nms

try:
    from maskdino import add_maskdino_config
except ImportError:
    print("❌ Errore: 'maskdino' non trovata. Esegui dalla root del progetto.")
    sys.exit(1)

try:
    from efficienzapannelli import chiedi_parametri_globali, calcola_diagnostica_dict
    HAS_EFFICIENCY = True
except ImportError:
    HAS_EFFICIENCY = False
    print("⚠️  efficienzapannelli.py non trovato — dati efficienza non disponibili")

try:
    from thermal_extractor import (get_ambient_temperature,
                                    extract_at_gps, pixels_to_area_m2)
    HAS_THERMAL = True
except ImportError:
    HAS_THERMAL = False
    print("⚠️  thermal_extractor.py non trovato — temperatura da DJI non disponibile")

try:
    from photo_matcher import load_drone_photos_index, find_best_photo
    HAS_MATCHER = True
except ImportError:
    HAS_MATCHER = False
    print("⚠️  photo_matcher.py non trovato — matching foto drone non disponibile")

# ==============================================================================
# ⚙️ CONFIGURAZIONE
# ==============================================================================
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR          = "/workspace/cortoswin"                         # cartella pesi training
WEIGHTS_PATH_OVERRIDE = os.path.join(os.path.dirname(BASE_DIR), "model_0001999.pth")
DATASET_FOLDER_NAME = "solar_datasets_nuoveannotazioni"

VAL_IMGS             = os.path.join(BASE_DIR, "inferenza_patches")   # patch tagliate per inferenza
DRONE_PHOTOS_DIR     = os.path.join(BASE_DIR, "foto_drone")          # foto raw DJI RJPEG
ORIGINAL_MOSAIC_PATH = os.path.join(BASE_DIR, "ortomosaico.tif")
TFW_PATH             = os.path.join(BASE_DIR, "ortomosaico.tfw")

OUTPUT_MOSAIC_MARKED  = os.path.join(BASE_DIR, "Mosaico_Finale_Rilevato.jpg")
OUTPUT_MOSAIC_DAMAGED = os.path.join(BASE_DIR, "Mosaico_Finale_Danneggiati.jpg")
OUTPUT_CSV            = os.path.join(BASE_DIR, "Rilevamenti_Pannelli.csv")

YAML_CONFIG = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation", "swin", "maskdino_swinl_bs16_50ep.yaml")
NUM_CLASSES = 2  # Pannello Danneggiato + Pannello Sano

SOGLIA_DETECTION       = 0.55
NMS_THRESH             = 0.40
MERGE_DIST_METERS      = 3
DOT_RADIUS_MOSAIC      = 8
BOOST                  = 2.5
SCORE_MIN_EXPORT       = 0.55
SCORE_MIN_DRAW_MOSAIC  = 0.65   # soglia minima per disegnare sul mosaico
AREA_MIN_DRAW_M2       = 0.4    # scarta falsi positivi piccoli (< 0.4 m²)
ASPECT_RATIO_MAX       = 5.0    # rapporto lati massimo (pannelli ≤ 5:1)

# Nuove Classi e Colori (BGR): sano=verde, danneggiato=rosso
CLASS_NAMES = {0: "Pannello Danneggiato", 1: "Pannello Sano"}
CLASS_COLORS = {
    0: (0, 0, 255),    # Danneggiato → Rosso
    1: (0, 200, 0),    # Sano        → Verde
}

# ==============================================================================
# 🛠️ FUNZIONI
# ==============================================================================

def get_best_weights(output_dir):
    best  = os.path.join(output_dir, "model_best.pth")
    final = os.path.join(output_dir, "model_final.pth")
    if os.path.exists(best):
        size = os.path.getsize(best) / 1e6
        print(f"✅ Modello: model_best.pth ({size:.1f} MB)")
        return best
    elif os.path.exists(final):
        size = os.path.getsize(final) / 1e6
        print(f"⚠️  Modello: model_final.pth ({size:.1f} MB)")
        return final
    else:
        checkpoints = glob.glob(os.path.join(output_dir, "model_*.pth"))
        if checkpoints:
            latest = max(checkpoints, key=os.path.getctime)
            print(f"⚠️  Uso: {os.path.basename(latest)}")
            return latest
        print(f"❌ Nessun .pth trovato in {output_dir}")
        sys.exit(1)


def get_geo_tools(tif_path):
    """
    Legge la matrice affine e calcola dinamicamente le coordinate GPS 
    (Latitudine, Longitudine) del centroide dell'ortomosaico per le chiamate API.
    """
    if not os.path.exists(tif_path):
        return None, None, None, None
    try:
        with rasterio.open(tif_path) as src:
            affine, crs = src.transform, src.crs
            w, h = src.width, src.height
            
        geo_transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        
        # Calcolo del centroide
        cx, cy = w / 2, h / 2
        proj_x, proj_y = affine * (cx, cy)
        lon_c, lat_c = geo_transformer.transform(proj_x, proj_y)
        
        return affine, geo_transformer, lat_c, lon_c
    except Exception as e:
        print(f"⚠️  Errore GeoTIFF: {e}")
        return None, None, None, None


def leggi_gsd(tif_path, tfw_path=None):
    if os.path.exists(tif_path):
        try:
            with rasterio.open(tif_path) as src:
                gsd_m = abs(src.transform.a)
            if 0.001 <= gsd_m <= 10.0:
                return gsd_m
        except Exception as e:
            print(f"⚠️  Errore lettura GSD da GeoTIFF: {e}")

    if tfw_path and os.path.exists(tfw_path):
        try:
            with open(tfw_path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            return abs(float(lines[0]))
        except Exception:
            pass
    return 0.01


def calcola_area_m2(mask, gsd_m):
    if gsd_m is None:
        return None
    n_pixel = int(np.sum(mask))
    return round(n_pixel * (gsd_m ** 2), 4)


def filtra_punti_unici(detections, soglia_metri):
    unique = []
    detections.sort(key=lambda x: x['score'], reverse=True)
    for p in detections:
        found = False
        for u in unique:
            dist = math.sqrt((p['lat'] - u['lat'])**2 + (p['lon'] - u['lon'])**2) * 111000
            if dist < soglia_metri:
                n = u['count']
                u['lat']   = (u['lat'] * n + p['lat'])   / (n + 1)
                u['lon']   = (u['lon'] * n + p['lon'])   / (n + 1)
                u['gx']    = (u['gx']  * n + p['gx'])    / (n + 1)
                u['gy']    = (u['gy']  * n + p['gy'])    / (n + 1)
                if u.get('t_panel') is not None and p.get('t_panel') is not None:
                    u['t_panel'] = (u['t_panel'] * n + p['t_panel']) / (n + 1)
                elif p.get('t_panel') is not None:
                    u['t_panel'] = p['t_panel']
                u['count'] += 1
                found = True
                break
        if not found:
            unique.append({**p, 'count': 1})
    return unique


def esporta_csv(panels, path):
    rows = []
    for i, p in enumerate(panels):
        eff = p.get('eff', {})
        rows.append({
            "id":              i + 1,
            "latitudine":      round(p['lat'], 8),
            "longitudine":     round(p['lon'], 8),
            "classe":          CLASS_NAMES.get(p.get('class_id', 1), "Pannello Sano"),
            "score":           round(p['score'], 4),
            "area_m2":         p.get('area_m2', None),
            "pixel_x":         int(p['gx']),
            "pixel_y":         int(p['gy']),
            "stato":           eff.get('stato', ''),
            "diagnosi":        eff.get('diagnosi', ''),
            "soh_pct":         eff.get('soh', ''),
            "t_reale_c":       eff.get('t_reale_hotspot', ''),
            "delta_t":         eff.get('delta_t', ''),
            "eta_pct":         eff.get('eta_hotspot_pct', ''),
            "p_erogata_w":     eff.get('p_erogata_w', ''),
            "p_persa_w":       eff.get('p_persa_w', ''),
            "perdita_euro":    eff.get('perdita_euro', ''),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"✅ CSV salvato: {path}  ({len(df)} righe)")



def setup_cfg(weights_path):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskdino_config(cfg)
    cfg.merge_from_file(YAML_CONFIG)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.MODEL.ROI_HEADS.NUM_CLASSES    = NUM_CLASSES
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.MaskDINO.NUM_CLASSES     = NUM_CLASSES

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST            = SOGLIA_DETECTION
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD    = 0.3

    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1600

    return cfg

# ==============================================================================
# 🚀 MAIN
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("  INFERENZA SOLAR PANELS — MaskDINO v4")
    print("  Classi: Pannello Sano / Pannello Danneggiato")
    print("="*60)

    # 1. Setup modello
    if os.path.exists(WEIGHTS_PATH_OVERRIDE):
        weights_path = WEIGHTS_PATH_OVERRIDE
        size = os.path.getsize(weights_path) / 1e6
        print(f"✅ Modello: {os.path.basename(weights_path)} ({size:.1f} MB)")
    else:
        weights_path = get_best_weights(OUTPUT_DIR)
    cfg          = setup_cfg(weights_path)
    predictor    = DefaultPredictor(cfg)

    device = cfg.MODEL.DEVICE.upper()
    if device == "CUDA":
        gpu_name = torch.cuda.get_device_name(0)
        vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    print(f"  Soglia detection: {SOGLIA_DETECTION:.0%} | NMS: {NMS_THRESH}")
    print(f"  Risoluzione: 800-1600px\n")


    # 2. Lettura GSD dal GeoTIFF
    gsd_m = leggi_gsd(ORIGINAL_MOSAIC_PATH, TFW_PATH)
    print(f"📐 GSD ortomosaico: {gsd_m*100:.2f} cm/pixel\n")

    # 3. Caricamento immagini
    all_files = sorted(
        glob.glob(os.path.join(VAL_IMGS, "*.jpg")) +
        glob.glob(os.path.join(VAL_IMGS, "*.JPG")) +
        glob.glob(os.path.join(VAL_IMGS, "*.png")) +
        glob.glob(os.path.join(VAL_IMGS, "*.PNG"))
    )
    if not all_files:
        print(f"❌ Nessuna immagine trovata in {VAL_IMGS}")
        sys.exit(1)

    print(f"📂 Trovate {len(all_files)} immagini totali.")
    scelta = input("👉 Quante immagini vuoi analizzare casualmente? (INVIO = tutte): ").strip()

    if scelta:
        try:
            limite    = min(int(scelta), len(all_files))
            all_files = random.sample(all_files, limite)
            print(f"🎲 Selezionate {len(all_files)} immagini casuali.\n")
        except ValueError:
            print("⚠️  Input non valido. Analizzo tutto il dataset.\n")
    else:
        print("🚀 Analizzo tutte le immagini.\n")

    # 4. Setup geo e calcolo centroide per API esterne
    affine, geo_transformer, lat_c, lon_c = get_geo_tools(ORIGINAL_MOSAIC_PATH)
    if not affine:
        print("⚠️  GeoTIFF non trovato — CSV e API meteo/solari saltati.\n")
    else:
        print(f"📍 Centroide Ortomosaico estratto: Lat {lat_c:.6f}, Lon {lon_c:.6f}\n")

    raw_detections = []

    # 4b. Indice foto drone e recupero Temperatura Ambiente
    photo_index = []
    t_amb_drone = 25.0   # default
    
    # CHIAMATA API METEO TRAMITE COORDINATE DEL CENTROIDE
    if lat_c is not None and lon_c is not None:
        try:
            # Nota: decommenta e usa la tua funzione meteo reale qui
            # from meteoclient import get_weather_data
            # t_amb_drone = get_weather_data(lat_c, lon_c)
            print(f"  🌤️  Coordinate Lat {lat_c:.4f}, Lon {lon_c:.4f} pronte per API Meteo.")
        except Exception as e:
            print(f"⚠️  Errore chiamata API Meteo: {e}")

    if HAS_THERMAL and HAS_MATCHER and os.path.isdir(DRONE_PHOTOS_DIR):
        print(f"📷 Indicizzazione foto drone in '{DRONE_PHOTOS_DIR}'...")
        photo_index = load_drone_photos_index(DRONE_PHOTOS_DIR)
        # Se disponibile, estrae T_amb dai metadati EXIF, sovrascrivendo l'API
        if photo_index:
            t_amb_drone = get_ambient_temperature(photo_index[0]["path"])
            print(f"   T_amb da EXIF drone: {t_amb_drone:.1f} °C")
    elif not os.path.isdir(DRONE_PHOTOS_DIR):
        print(f"⚠️  Cartella foto drone non trovata: {DRONE_PHOTOS_DIR}")
        print(f"     Temperatura estratta dai metadati non disponibile.")

    # 5. Loop inferenza
    total_panels = 0
    for i, path in enumerate(all_files):
        filename = os.path.basename(path)
        match    = re.search(r"tile_col_(\d+)_row_(\d+)", filename)
        off_x, off_y = (int(match.group(1)), int(match.group(2))) if match else (0, 0)

        img = cv2.imread(path)
        if img is None:
            print(f"  ⚠️  Impossibile leggere: {filename}")
            continue

        outputs   = predictor(img)
        instances = outputs["instances"].to("cpu")

        if len(instances) > 0:
            instances._fields['scores'] = torch.clamp(instances.scores * BOOST, max=1.0)

        if len(instances) > 0:
            score_mask = instances.scores >= SOGLIA_DETECTION
            instances  = instances[score_mask]

        if len(instances) > 0:
            keep      = nms(instances.pred_boxes.tensor, instances.scores, NMS_THRESH)
            instances = instances[keep]

        num_panels = len(instances)
        max_score  = instances.scores.max().item() if num_panels > 0 else 0.0
        total_panels += num_panels

        print(f"📸 [{i+1:3d}/{len(all_files)}] {filename:<45} | "
              f"Pannelli: {num_panels:2d} | Conf: {max_score:.1%}")

        if num_panels == 0:
            continue

        if affine and geo_transformer:
            masks    = instances.pred_masks.numpy()
            boxes    = instances.pred_boxes.tensor.numpy()
            scores   = instances.scores.numpy()
            classes  = instances.pred_classes.numpy() if instances.has("pred_classes") else np.zeros(num_panels, dtype=int)

            _ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            _ker_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            for k in range(num_panels):
                patch_mask = masks[k].astype(bool)
                mask_u8    = patch_mask.astype("uint8") * 255

                # Pulizia maschera e rettangolo ruotato minimo
                mask_clean = cv2.morphologyEx(mask_u8,    cv2.MORPH_CLOSE, _ker_close, iterations=2)
                mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN,  _ker_open,  iterations=1)
                contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours    = [c for c in contours if cv2.contourArea(c) > 200]

                global_box_pts = None
                contour_local  = None
                if contours:
                    c_big = max(contours, key=cv2.contourArea)

                    # Filtro aspect ratio: scarta oggetti troppo allungati o quadrati anomali
                    rect     = cv2.minAreaRect(c_big)
                    rw, rh   = rect[1]
                    if min(rw, rh) > 0:
                        ar = max(rw, rh) / min(rw, rh)
                        if ar > ASPECT_RATIO_MAX:
                            continue   # non è un pannello

                    M = cv2.moments(c_big)
                    if M["m00"] != 0:
                        cx_local = int(M["m10"] / M["m00"])
                        cy_local = int(M["m01"] / M["m00"])
                    else:
                        cx_local = int((boxes[k][0] + boxes[k][2]) / 2)
                        cy_local = int((boxes[k][1] + boxes[k][3]) / 2)

                    local_box      = np.int32(cv2.boxPoints(rect))
                    global_box_pts = local_box.copy()
                    global_box_pts[:, 0] += off_x
                    global_box_pts[:, 1] += off_y
                    contour_local  = c_big   # contorno in coordinate patch locale
                else:
                    cx_local = int((boxes[k][0] + boxes[k][2]) / 2)
                    cy_local = int((boxes[k][1] + boxes[k][3]) / 2)

                gx, gy = off_x + cx_local, off_y + cy_local
                proj_x, proj_y = affine * (gx, gy)
                lon, lat = geo_transformer.transform(proj_x, proj_y)

                t_panel    = None
                area_m2    = calcola_area_m2(patch_mask, gsd_m)
                class_id_k = int(classes[k])

                if HAS_THERMAL and HAS_MATCHER and photo_index:
                    best_photo = find_best_photo(lat, lon, photo_index)
                    if best_photo:
                        # Modalità estrazione: max hotspot per danneggiato, media per sano
                        mode_t = "max" if class_id_k == 0 else "mean"

                        # Stima raggio termica dalla dimensione della mask nell'ortomosaico:
                        # area_mask [m²] → raggio equivalente nel sensore termico (~5cm/px)
                        n_px    = int(np.sum(patch_mask))
                        area_m2 = pixels_to_area_m2(n_px, gsd_m)
                        drone_gsd_est = 0.05   # ~5cm/px DJI H30T a 30m
                        r_thermal = max(10, int(
                            math.sqrt((n_px * gsd_m**2) / math.pi) / drone_gsd_est
                        ))
                        r_thermal = min(r_thermal, 45)  # cap massimo

                        # Interroga SDK per temperatura usando la posizione GPS del centroide
                        t_panel = extract_at_gps(
                            best_photo, lat, lon,
                            mode=mode_t,
                            radius_px=r_thermal,
                        )

                        # [DEBUG] Stampa di controllo pannello ↔ foto drone
                        classe_debug = CLASS_NAMES.get(class_id_k, "?")
                        t_label = "T_max" if class_id_k == 0 else "T_media"
                        t_str   = f"{t_panel:.1f} °C" if t_panel is not None else "N/D"
                        print(f"  [DEBUG] Pannello #{len(raw_detections)+1} ({classe_debug}) "
                              f"GPS ({lat:.6f}, {lon:.6f}) → "
                              f"foto: {os.path.basename(best_photo)} | "
                              f"{t_label}={t_str}  r={r_thermal}px")


                raw_detections.append({
                    'lat':           lat,
                    'lon':           lon,
                    'gx':            gx,
                    'gy':            gy,
                    'score':         float(scores[k]),
                    'class_id':      class_id_k,
                    'area_m2':       area_m2,
                    'box_pts':       global_box_pts,
                    't_panel':       t_panel,
                    'patch_path':    path,            # patch originale (per debug)
                    'contour_local': contour_local,   # contorno in coord patch (per debug)
                })

    print(f"\n{'─'*60}")
    print(f"  Pannelli rilevati (pre-fusione): {total_panels}")

    if not raw_detections:
        print("❌ Nessun pannello geolocalizzato.")
        sys.exit(0)

    # 6b. Parametri efficienza e PVGIS (PRIMA del merge — serve la diagonale pannello)
    eff_params = None
    if HAS_EFFICIENCY:
        print(f"\n{'─'*60}")

        # CHIAMATA API PVGIS TRAMITE COORDINATE DEL CENTROIDE
        pvgis_data = None
        if lat_c is not None and lon_c is not None:
            try:
                from pvgclient import get_pvgis_data
                pvgis_data = get_pvgis_data(lat_c, lon_c)
            except Exception as e:
                print(f"⚠️  Errore recupero API PVGIS: {e}")

        # Chiede tipo pannello + dimensioni fisiche
        eff_params = chiedi_parametri_globali()

        # Inietta i dati scaricati nei parametri globali
        if pvgis_data:
            eff_params['esh'] = pvgis_data['esh']
            eff_params['giorni_utili'] = pvgis_data['giorni_utili']

        # Temperatura ambiente automatica
        eff_params['t_amb'] = t_amb_drone

    # 6. Fusione duplicati — soglia dinamica basata sulla diagonale del pannello
    if eff_params and eff_params.get('w_mod') and eff_params.get('h_mod'):
        w_mod = eff_params['w_mod']
        h_mod = eff_params['h_mod']
        # Diagonale pannello × 0.9: fonde doppioni sullo stesso pannello
        # senza toccare pannelli adiacenti (distanza tipica > 0.1m)
        merge_dist = round(math.sqrt(w_mod**2 + h_mod**2) * 0.9, 3)
        print(f"\n📐 Merge dinamico: diag({w_mod}m×{h_mod}m)×0.9 = {merge_dist:.2f} m")
    else:
        merge_dist = MERGE_DIST_METERS
        print(f"\n📐 Merge fisso: {merge_dist} m (dimensioni pannello non disponibili)")

    print(f"🔄 Fusione duplicati (soglia: {merge_dist} m)...")
    final_panels = filtra_punti_unici(raw_detections, merge_dist)
    dopo_merge = len(final_panels)
    print(f"  Dopo merge: {dopo_merge}")

    # Filtro score
    final_panels = [p for p in final_panels if p['score'] >= SCORE_MIN_EXPORT]
    print(f"  Dopo filtro score >= {SCORE_MIN_EXPORT}: {len(final_panels)}")

    # Filtro area: scarta solo i veri falsi positivi (<0.1 m²)
    # None area viene tenuta (foto drone non disponibile non è motivo di scarto)
    final_panels = [p for p in final_panels
                    if p.get('area_m2') is None or p.get('area_m2', 0) >= 0.1]
    print(f"  Dopo filtro area >= 0.1 m² (None=tenuto): {len(final_panels)}")
    print(f"  Pannelli unici finali: {len(final_panels)}")

    # T_sano: calcolato dopo il merge (serve final_panels)
    t_sano_is_fallback = False
    if eff_params is not None:
        temp_sani = [
            p['t_panel'] for p in final_panels
            if p.get('class_id') == 1 and p.get('t_panel') is not None
        ]
        if temp_sani:
            eff_params['t_sano'] = max(temp_sani)
            print(f"  T_sano baseline (sano più caldo): {eff_params['t_sano']:.1f} °C  "
                  f"[su {len(temp_sani)} pannelli sani]")
        elif eff_params.get('t_sano') is None:
            eff_params['t_sano'] = t_amb_drone + 10.0
            t_sano_is_fallback   = True
            print(f"  ⚠️  T_sano non trovata — fallback: {eff_params['t_sano']:.1f} °C  [!! sul mosaico]")

    # 7. Calcolo efficienza per ogni pannello
    if HAS_EFFICIENCY and eff_params:
        print(f"\n{'─'*60}")
        print("  Calcolo efficienza per ogni pannello...")
        for p in final_panels:
            t_hot = p.get('t_panel')
            if t_hot is None:
                # Fallback basato sulla classe: danneggiato simula +15°C hotspot
                if p.get('class_id') == 0:
                    t_hot = eff_params['t_sano'] + 15.0
                else:
                    t_hot = eff_params['t_sano']
            try:
                p['eff'] = calcola_diagnostica_dict(t_hot, p.get('area_m2', 1.6), eff_params)
            except Exception as e:
                print(f"  ⚠️  Errore calcolo efficienza pannello: {e}")
                p['eff'] = {}

    # 8. CSV
    esporta_csv(final_panels, OUTPUT_CSV)

    # 10. Mosaici annotati (tutti i pannelli + solo danneggiati)
    if os.path.exists(ORIGINAL_MOSAIC_PATH):
        try:
            mosaico_base = cv2.imread(ORIGINAL_MOSAIC_PATH)
            if mosaico_base is not None:
                mosaico_all = mosaico_base.copy()
                mosaico_dmg = mosaico_base.copy()
                cnt_all = 0
                cnt_dmg = 0

                font       = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness  = 1

                da_disegnare = sum(1 for p in final_panels if p['score'] >= SCORE_MIN_DRAW_MOSAIC)
                print(f"  Da disegnare sul mosaico (score >= {SCORE_MIN_DRAW_MOSAIC}): {da_disegnare}/{len(final_panels)}")

                for idx, p in enumerate(final_panels):
                    if p['score'] < SCORE_MIN_DRAW_MOSAIC:
                        continue
                    if (p.get('area_m2') is not None and
                            p.get('area_m2', 0) < AREA_MIN_DRAW_M2):
                        continue   # scarta falsi positivi piccoli

                    class_id    = p.get('class_id', 1)
                    classe_nome = CLASS_NAMES.get(class_id, "Pannello Sano")
                    color       = CLASS_COLORS.get(class_id, (0, 200, 0))
                    cx, cy      = int(p['gx']), int(p['gy'])

                    eff_pct = p.get('eff', {}).get('eta_hotspot_pct')
                    warn    = " !!" if t_sano_is_fallback else ""
                    if eff_pct is not None:
                        label = f"#{idx+1} η={eff_pct:.1f}%{warn}"
                    else:
                        if classe_nome == "Pannello Danneggiato":
                            label = f"#{idx+1} DANN.{warn}"
                        else:
                            label = f"#{idx+1} SANO"

                    def _draw_panel(canvas, box_pts, cx, cy, color, label):
                        if box_pts is not None:
                            cv2.drawContours(canvas, [box_pts], 0, color, 3)
                        else:
                            cv2.circle(canvas, (cx, cy), DOT_RADIUS_MOSAIC, color, -1)
                        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        tx = max(0, cx - tw // 2)
                        ty = max(th + 4, cy - 14)
                        cv2.rectangle(canvas, (tx-2, ty-th-2), (tx+tw+2, ty+2), (0, 0, 0), -1)
                        cv2.putText(canvas, label, (tx, ty),
                                    font, font_scale, color, thickness, cv2.LINE_AA)

                    # Mosaico completo (tutti)
                    _draw_panel(mosaico_all, p.get('box_pts'), cx, cy, color, label)
                    cnt_all += 1

                    # Mosaico solo danneggiati
                    if class_id == 0:
                        _draw_panel(mosaico_dmg, p.get('box_pts'), cx, cy, color, label)
                        cnt_dmg += 1

                cv2.imwrite(OUTPUT_MOSAIC_MARKED,  mosaico_all)
                cv2.imwrite(OUTPUT_MOSAIC_DAMAGED, mosaico_dmg)
                print(f"✅ Mosaico completo: {OUTPUT_MOSAIC_MARKED}  ({cnt_all} pannelli)")
                print(f"✅ Mosaico danneggiati: {OUTPUT_MOSAIC_DAMAGED}  ({cnt_dmg} pannelli)")
        except Exception as e:
            print(f"⚠️  Errore mosaico: {e}")

    # ── DEBUG PATCH post-mosaico (prime 5) ──────────────────────────────────────
    # Usa la patch originale del mosaico: la mask è già nelle coordinate giuste
    print(f"\n{'─'*60}")
    print("  📸 Generazione immagini debug su patch (prime 5)...")
    dbg_count = 0
    for p_d in final_panels:
        if dbg_count >= 5:
            break
        if p_d['score'] < SCORE_MIN_DRAW_MOSAIC:
            continue
        patch_p  = p_d.get('patch_path')
        cnt_loc  = p_d.get('contour_local')
        if patch_p is None or cnt_loc is None or not os.path.exists(patch_p):
            continue

        try:
            patch_img = cv2.imread(patch_p)
            if patch_img is None:
                continue

            cid_d   = p_d.get('class_id', 1)
            color_d = CLASS_COLORS.get(cid_d, (0, 200, 0))

            # Fill semi-trasparente con il contorno della mask AI
            overlay = patch_img.copy()
            cv2.fillPoly(overlay, [cnt_loc], color_d)
            cv2.addWeighted(overlay, 0.30, patch_img, 0.70, 0, patch_img)
            cv2.polylines(patch_img, [cnt_loc], True, color_d, 2)

            # Centroide
            M = cv2.moments(cnt_loc)
            if M["m00"] != 0:
                cx_d = int(M["m10"] / M["m00"])
                cy_d = int(M["m01"] / M["m00"])
            else:
                cx_d = patch_img.shape[1] // 2
                cy_d = patch_img.shape[0] // 2
            cv2.circle(patch_img, (cx_d, cy_d), 3, (255, 255, 255), -1)

            # Temperatura
            t_val  = p_d.get('t_panel')
            t_s    = f"{t_val:.1f}C" if t_val is not None else "N/D"
            nome_d = "DANN." if cid_d == 0 else "SANO"
            eff_d  = p_d.get('eff', {}).get('eta_hotspot_pct')
            if eff_d is not None:
                lbl = f"#{dbg_count+1} {nome_d} η={eff_d:.1f}% T={t_s}"
            else:
                lbl = f"#{dbg_count+1} {nome_d} T={t_s}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th_), _ = cv2.getTextSize(lbl, font, 0.55, 2)
            lx = max(2, cx_d - tw // 2)
            ly = max(th_ + 4, cy_d - 20)
            cv2.rectangle(patch_img, (lx-2, ly-th_-4), (lx+tw+2, ly+4), (0, 0, 0), -1)
            cv2.putText(patch_img, lbl, (lx, ly), font, 0.55, color_d, 2, cv2.LINE_AA)

            dbg_count += 1
            out_d = os.path.join(BASE_DIR, f"debug_patch_{dbg_count:02d}.jpg")
            cv2.imwrite(out_d, patch_img)
            print(f"  📸 [{dbg_count}/5] {out_d}")

        except Exception as e:
            print(f"  ⚠️  Debug patch {dbg_count+1}: {e}")

    print(f"\n🎯 PANNELLI SOLARI RILEVATI: {len(final_panels)}")
    print(f"📁 Output generati:")
    print(f"   - {OUTPUT_CSV}")
    print(f"   - {OUTPUT_MOSAIC_MARKED}  (tutti i pannelli)")
    print(f"   - {OUTPUT_MOSAIC_DAMAGED}  (solo danneggiati)")
    print("🎯 PROCEDURA COMPLETATA.\n")


if __name__ == "__main__":
    main()
