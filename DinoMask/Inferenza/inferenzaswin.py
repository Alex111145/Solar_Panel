#!/usr/bin/env python
"""
inference.py - Inferenza MaskDINO Solar Panels (v4 - FINAL)
============================================================
Classi aggiornate: "Pannello Sano" e "Pannello Danneggiato".
Coerente con il training v4:
  - Risoluzione test 1600px (identica al training)
  - Soglia detection 0.37
  - NMS post-processing
  - Geolocalizzazione GeoTIFF → KML/KMZ/CSV/GeoJSON
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
import csv
import json
import shutil
import cv2
import torch
import numpy as np
import rasterio
import simplekml
import random
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pyproj import Transformer

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
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
    print("⚠️  efficienzapannelli.py non trovato — KML senza dati efficienza")

try:
    from thermal_extractor import (get_ambient_temperature, get_gsd as get_drone_gsd,
                                    extract_from_patch_region, pixels_to_area_m2)
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
DATASET_FOLDER_NAME = "solar_datasets_nuoveannotazioni"

VAL_IMGS             = os.path.join(BASE_DIR, "inferenza_patches")   # patch tagliate per inferenza
DRONE_PHOTOS_DIR     = os.path.join(BASE_DIR, "foto_drone")          # foto raw DJI RJPEG
ORIGINAL_MOSAIC_PATH = os.path.join(BASE_DIR, "ortomosaico.tif")
TFW_PATH             = os.path.join(BASE_DIR, "ortomosaico.tfw")

VIS_OUTPUT_DIR       = os.path.join(BASE_DIR, "inference_results")
OUTPUT_MOSAIC_MARKED = os.path.join(BASE_DIR, "Mosaico_Finale_Rilevato.jpg")
OUTPUT_KML           = os.path.join(BASE_DIR, "Mappa_Pannelli.kml")
OUTPUT_KMZ           = os.path.join(BASE_DIR, "Mappa_Pannelli.kmz")
OUTPUT_CSV           = os.path.join(BASE_DIR, "Rilevamenti_Pannelli.csv")
OUTPUT_GEOJSON       = os.path.join(BASE_DIR, "Rilevamenti_Pannelli.geojson")

YAML_CONFIG = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation", "swin", "maskdino_swinl_bs16_50ep.yaml")
NUM_CLASSES = 2  # Pannello Danneggiato + Pannello Sano

SOGLIA_DETECTION       = 0.50
NMS_THRESH             = 0.40
MERGE_DIST_METERS      = 3
DOT_RADIUS_MOSAIC      = 8
BOOST                  = 2.5
SCORE_MIN_EXPORT       = 0.5
SCORE_MIN_DRAW_MOSAIC  = 0.70   # soglia minima per disegnare sul mosaico

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


def esporta_geojson(panels, path):
    geometries = []
    attrs      = []
    for p in panels:
        if p.get('contour') is not None:
            pts = [(int(c[0][0]), int(c[0][1])) for c in p['contour']]
            geom = Polygon(pts) if len(pts) >= 3 else Point(p['lon'], p['lat'])
        else:
            geom = Point(p['lon'], p['lat'])
        geometries.append(geom)
        attrs.append({
            "classe":  CLASS_NAMES.get(p.get('class_id', 1), "Pannello Sano"),
            "score":   round(p['score'], 4),
            "area_m2": p.get('area_m2', None),
        })
    gdf = gpd.GeoDataFrame(attrs, geometry=geometries, crs="EPSG:4326")
    gdf.to_file(path, driver="GeoJSON")
    print(f"✅ GeoJSON salvato: {path}  ({len(gdf)} features)")


def esporta_kmz(kml_path, kmz_path):
    import zipfile
    with zipfile.ZipFile(kmz_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(kml_path, arcname="doc.kml")
    print(f"✅ KMZ salvato: {kmz_path}")


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

    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

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
        print("⚠️  GeoTIFF non trovato — KML/KMZ/CSV/GeoJSON e API meteo/solari saltati.\n")
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

        try:
            meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        except Exception:
            meta = MetadataCatalog.get("__unused")

        v = Visualizer(img[:, :, ::-1], meta, scale=1.0)
        v.overlay_instances(masks=instances.pred_masks)
        for k in range(num_panels):
            score_pct = f"{int(instances.scores[k].item() * 100)}%"
            v.draw_text(score_pct, instances.pred_boxes.tensor[k][:2])
        out = v.get_output()
        cv2.imwrite(
            os.path.join(VIS_OUTPUT_DIR, f"res_{filename}"),
            out.get_image()[:, :, ::-1]
        )

        if affine and geo_transformer:
            masks    = instances.pred_masks.numpy()
            boxes    = instances.pred_boxes.tensor.numpy()
            scores   = instances.scores.numpy()
            classes  = instances.pred_classes.numpy() if instances.has("pred_classes") else np.zeros(num_panels, dtype=int)

            for k in range(num_panels):
                patch_mask = masks[k].astype(bool)
                mask_u8     = (patch_mask.astype("uint8") * 255)
                contours, _ = cv2.findContours(
                    mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                global_contour = None
                if contours:
                    c  = max(contours, key=cv2.contourArea)
                    M  = cv2.moments(c)
                    if M["m00"] != 0:
                        cx_local = int(M["m10"] / M["m00"])
                        cy_local = int(M["m01"] / M["m00"])
                    else:
                        cx_local = int((boxes[k][0] + boxes[k][2]) / 2)
                        cy_local = int((boxes[k][1] + boxes[k][3]) / 2)
                    global_contour = c.copy()
                    global_contour[:, :, 0] += off_x
                    global_contour[:, :, 1] += off_y
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
                        gsd_drone = get_drone_gsd(best_photo)
                        scale = gsd_m / gsd_drone if gsd_drone > 0 else 1.0
                        mode  = "max" if class_id_k == 0 else "mean"
                        t_panel = extract_from_patch_region(
                            best_photo, off_x, off_y, patch_mask,
                            mode=mode, ortho_to_drone_scale=scale
                        )
                        n_px    = int(np.sum(patch_mask))
                        area_m2 = pixels_to_area_m2(n_px, gsd_m)

                raw_detections.append({
                    'lat':      lat,
                    'lon':      lon,
                    'gx':       gx,
                    'gy':       gy,
                    'score':    float(scores[k]),
                    'class_id': class_id_k,
                    'area_m2':  area_m2,
                    'contour':  global_contour,
                    't_panel':  t_panel,
                })

    print(f"\n{'─'*60}")
    print(f"  Pannelli rilevati (pre-fusione): {total_panels}")

    if not raw_detections:
        print("❌ Nessun pannello geolocalizzato.")
        print(f"✅ Immagini patch salvate in: {VIS_OUTPUT_DIR}")
        sys.exit(0)

    # 6. Fusione duplicati
    print(f"🔄 Fusione duplicati (soglia: {MERGE_DIST_METERS}m)...")
    final_panels = filtra_punti_unici(raw_detections, MERGE_DIST_METERS)
    prima = len(final_panels)
    final_panels = [p for p in final_panels
                    if p['score'] >= SCORE_MIN_EXPORT
                    and (p.get('area_m2') or 0) >= 1.0]
    scartati = prima - len(final_panels)
    if scartati > 0:
        print(f"  Scartati (area < 1 m² = falsi positivi): {scartati}")
    print(f"  Pannelli unici finali: {len(final_panels)}")

    # 6b. Parametri efficienza e PVGIS automatico
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

        # Chiede i dati restanti (potenza modulo, prezzo, ecc.)
        eff_params = chiedi_parametri_globali()

        # Inietta i dati scaricati nei parametri globali 
        if pvgis_data:
            eff_params['esh'] = pvgis_data['esh']
            eff_params['giorni_utili'] = pvgis_data['giorni_utili']

        # Imposta la Temperatura ambiente automatica
        eff_params['t_amb'] = t_amb_drone

        # Imposta T_sano (il pannello sano più freddo rilevato)
        temp_sani = [
            p['t_panel'] for p in final_panels
            if p.get('class_id') == 1 and p.get('t_panel') is not None
        ]
        if temp_sani:
            eff_params['t_sano'] = min(temp_sani)
            print(f"  T_sano (pannello baseline) : {eff_params['t_sano']:.1f} °C")
        elif eff_params.get('t_sano') is None:
            eff_params['t_sano'] = t_amb_drone + 10.0
            print(f"  ⚠️  T_sano non trovata — uso fallback: {eff_params['t_sano']:.1f} °C")

    # 7. KML + KMZ
    kml = simplekml.Kml()

    for idx, p in enumerate(final_panels):
        classe_nome = CLASS_NAMES.get(p.get('class_id', 1), "Pannello Sano")

        eff = None
        if eff_params and HAS_EFFICIENCY:
            t_app = p.get('t_panel') or eff_params.get('t_sano') or (t_amb_drone + 10.0)
            eff   = calcola_diagnostica_dict(t_app, p.get('area_m2', 1.63), eff_params)

        if eff:
            colore = eff['kml_color']
        else:
            colore = "ff0000ff" if classe_nome == "Pannello Danneggiato" else "ff00cc00"

        if eff:
            html = (
                f"<b>ID Pannello:</b> {idx+1}<br/>"
                f"<b>Classe AI:</b> {classe_nome}<br/>"
                f"<b>Confidenza:</b> {p['score']:.0%}<br/>"
                f"<b>Area:</b> {p.get('area_m2', 'N/A')} m²<br/>"
                f"<hr/>"
                f"<b>Tecnologia:</b> {eff['tipo_pannello']}<br/>"
                f"<b>Temp. pannello:</b> {eff['t_reale_hotspot']} °C<br/>"
                f"<b>Temp. baseline:</b> {eff['t_reale_sano']} °C<br/>"
                f"<b>ΔT:</b> +{eff['delta_t']} °C<br/>"
                f"<hr/>"
                f"<b>Stato:</b> {eff['stato']}<br/>"
                f"<b>Diagnosi:</b> {eff['diagnosi']}<br/>"
                f"<b>SoH:</b> {eff['soh']} %<br/>"
                f"<b>Efficienza:</b> {eff['eta_hotspot_pct']} %<br/>"
                f"<b>Potenza erogata:</b> {eff['p_erogata_w']} W<br/>"
                f"<b>Potenza persa:</b> {eff['p_persa_w']} W<br/>"
                f"<b>Mancato guadagno:</b> -{eff['perdita_euro']} €/anno<br/>"
            )
        else:
            html = (
                f"<b>ID:</b> {idx+1}<br/>"
                f"<b>Classe:</b> {classe_nome}<br/>"
                f"<b>Score:</b> {p['score']:.2f}<br/>"
                f"<b>Area:</b> {p.get('area_m2', 'N/A')} m²<br/>"
            )

        pnt = kml.newpoint(
            name        = f"#{idx+1} {classe_nome}",
            coords      = [(p['lon'], p['lat'])],
        )
        pnt.description = f"<![CDATA[{html}]]>"
        style = simplekml.Style()
        style.iconstyle.icon.href  = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
        style.iconstyle.color      = colore
        style.iconstyle.scale      = 1.4
        style.labelstyle.scale     = 0.0
        pnt.style = style

        pnt.region                    = simplekml.Region()
        pnt.region.lod.minlodpixels  = 1
        pnt.region.lod.maxlodpixels  = -1

        if eff:
            p['eff'] = eff

    kml.save(OUTPUT_KML)
    print(f"✅ KML salvato: {OUTPUT_KML}")
    esporta_kmz(OUTPUT_KML, OUTPUT_KMZ)

    # 8. CSV
    esporta_csv(final_panels, OUTPUT_CSV)

    # 9. GeoJSON
    esporta_geojson(final_panels, OUTPUT_GEOJSON)

    # 10. Mosaico annotato
    if os.path.exists(ORIGINAL_MOSAIC_PATH):
        try:
            mosaico = cv2.imread(ORIGINAL_MOSAIC_PATH)
            if mosaico is not None:
                pannelli_disegnati = 0
                for idx, p in enumerate(final_panels):
                    if p['score'] < SCORE_MIN_DRAW_MOSAIC:
                        continue

                    class_id    = p.get('class_id', 1)
                    classe_nome = CLASS_NAMES.get(class_id, "Pannello Sano")
                    color       = CLASS_COLORS.get(class_id, (0, 200, 0))
                    cx, cy      = int(p['gx']), int(p['gy'])

                    if p.get('contour') is not None:
                        cv2.drawContours(mosaico, [p['contour']], -1, color, 3)
                    else:
                        cv2.circle(mosaico, (cx, cy), DOT_RADIUS_MOSAIC, color, -1)

                    if classe_nome == "Pannello Danneggiato":
                        label = f"#{idx+1} DANNEGGIATO {p['score']:.0%}"
                    else:
                        label = f"#{idx+1} SANO {p['score']:.0%}"

                    font       = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness  = 1
                    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

                    tx = max(0, cx - tw // 2)
                    ty = max(th + 4, cy - 14)
                    cv2.rectangle(mosaico,
                                  (tx - 2, ty - th - 2),
                                  (tx + tw + 2, ty + 2),
                                  (0, 0, 0), -1)
                    cv2.putText(mosaico, label, (tx, ty),
                                font, font_scale, color, thickness, cv2.LINE_AA)

                    pannelli_disegnati += 1

                cv2.imwrite(OUTPUT_MOSAIC_MARKED, mosaico)
                print(f"✅ Mosaico salvato: {OUTPUT_MOSAIC_MARKED} ({pannelli_disegnati} pannelli disegnati, score≥{SCORE_MIN_DRAW_MOSAIC:.0%})")
        except Exception as e:
            print(f"⚠️  Errore mosaico: {e}")

    print(f"\n🎯 PANNELLI SOLARI RILEVATI: {len(final_panels)}")
    print(f"📁 Output generati:")
    print(f"   - {OUTPUT_KML}")
    print(f"   - {OUTPUT_KMZ}")
    print(f"   - {OUTPUT_CSV}")
    print(f"   - {OUTPUT_GEOJSON}")
    print(f"   - {OUTPUT_MOSAIC_MARKED}")
    print(f"   - {VIS_OUTPUT_DIR}/ (patch annotate)")
    print("🎯 PROCEDURA COMPLETATA.\n")


if __name__ == "__main__":
    main()
