#!/usr/bin/env python
"""
inference.py - Inferenza MaskDINO Solar Panels (v5 - ResNet-50)
================================================================
Coerente con il training v5 (R50):
  - OUTPUT_DIR: output_solar_r50
  - YAML: maskdino_R50_bs16_50ep_3s.yaml
  - Risoluzione test 800-1600px
  - Soglia detection 0.50
  - NMS post-processing
  - Geolocalizzazione GeoTIFF → KML/KMZ/CSV/GeoJSON
  - Fusione duplicati per distanza
  - Calcolo area degradata (GSD²)
  - Selezione casuale N immagini
  - Mosaico annotato con contorni colorati per classe
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

# ==============================================================================
# ⚙️ CONFIGURAZIONE — allineata con train.py v5 (R50)
# ==============================================================================
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR          = os.path.join(BASE_DIR, "output_solar_r50")          # ← allineato con train v5
DATASET_FOLDER_NAME = "solar_datasets_nuoveannotazioni"

VAL_IMGS             = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "test")
ORIGINAL_MOSAIC_PATH = os.path.join(BASE_DIR, "ortomosaico.tif")
TFW_PATH             = os.path.join(BASE_DIR, "ortomosaico.tfw")

VIS_OUTPUT_DIR       = os.path.join(BASE_DIR, "inference_results")
OUTPUT_MOSAIC_MARKED = "Mosaico_Finale_Rilevato.jpg"
OUTPUT_KML           = "Mappa_Pannelli.kml"
OUTPUT_KMZ           = "Mappa_Pannelli.kmz"
OUTPUT_CSV           = "Rilevamenti_Pannelli.csv"
OUTPUT_GEOJSON       = "Rilevamenti_Pannelli.geojson"

YAML_CONFIG = os.path.join(                                                # ← allineato con train v5
    BASE_DIR, "configs", "coco", "instance-segmentation",
    "maskdino_R50_bs16_50ep_3s.yaml"
)
NUM_CLASSES = 1

SOGLIA_DETECTION  = 0.50
NMS_THRESH        = 0.40
MERGE_DIST_METERS = 1
DOT_RADIUS_MOSAIC = 8
BOOST             = 2.5
SCORE_MIN_EXPORT  = 0.5

# Colori per classe (BGR): integro=verde, hotspot=rosso, degrado=arancione
CLASS_COLORS = {
    0: (0, 200, 0),    # PV_Module  → verde
    1: (0, 0, 255),    # Hotspot    → rosso
    2: (0, 140, 255),  # Degrado    → arancione
}
CLASS_NAMES = {0: "PV_Module", 1: "Hotspot", 2: "Degrado"}

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
    if not os.path.exists(tif_path):
        return None, None
    try:
        with rasterio.open(tif_path) as src:
            affine, crs = src.transform, src.crs
        return affine, Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    except Exception as e:
        print(f"⚠️  Errore GeoTIFF: {e}")
        return None, None


def leggi_gsd(tfw_path):
    """Legge il GSD dal file TFW (coefficiente A = pixel_size_x in gradi)."""
    if not os.path.exists(tfw_path):
        return None
    try:
        with open(tfw_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        # Conversione gradi → metri (approssimazione equatoriale: 1° ≈ 111320 m)
        gsd_deg = abs(float(lines[0]))
        gsd_m   = gsd_deg * 111320
        return gsd_m
    except Exception as e:
        print(f"⚠️  Errore lettura TFW: {e}")
        return None


def calcola_area_m2(mask, gsd_m):
    """Calcola l'area reale della maschera: N_pixel * GSD²."""
    if gsd_m is None:
        return None
    n_pixel = int(np.sum(mask))
    return round(n_pixel * (gsd_m ** 2), 4)


def filtra_punti_unici(detections, soglia_metri):
    """Fonde rilevamenti multipli dello stesso pannello fisico."""
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
                u['count'] += 1
                found = True
                break
        if not found:
            unique.append({**p, 'count': 1})
    return unique


def esporta_csv(panels, path):
    """Esporta il database in CSV con pandas."""
    rows = []
    for p in panels:
        rows.append({
            "latitudine":  round(p['lat'], 8),
            "longitudine": round(p['lon'], 8),
            "classe":      CLASS_NAMES.get(p.get('class_id', 0), "PV_Module"),
            "score":       round(p['score'], 4),
            "area_m2":     p.get('area_m2', None),
            "pixel_x":     int(p['gx']),
            "pixel_y":     int(p['gy']),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"✅ CSV salvato: {path}  ({len(df)} righe)")


def esporta_geojson(panels, path):
    """Esporta il database in GeoJSON con geopandas (poligoni o punti)."""
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
            "classe":  CLASS_NAMES.get(p.get('class_id', 0), "PV_Module"),
            "score":   round(p['score'], 4),
            "area_m2": p.get('area_m2', None),
        })
    gdf = gpd.GeoDataFrame(attrs, geometry=geometries, crs="EPSG:4326")
    gdf.to_file(path, driver="GeoJSON")
    print(f"✅ GeoJSON salvato: {path}  ({len(gdf)} features)")


def esporta_kmz(kml_path, kmz_path):
    """Comprimi il KML in KMZ."""
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

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST         = SOGLIA_DETECTION
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.3

    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1600

    return cfg

# ==============================================================================
# 🚀 MAIN
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("  INFERENZA SOLAR PANELS — MaskDINO v5 (ResNet-50)")
    print("="*60)

    # Verifica YAML
    if not os.path.exists(YAML_CONFIG):
        print(f"❌ YAML non trovato: {YAML_CONFIG}")
        sys.exit(1)

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

    # 2. Lettura GSD dal TFW
    gsd_m = leggi_gsd(TFW_PATH)
    if gsd_m:
        print(f"📐 GSD rilevato: {gsd_m*100:.2f} cm/pixel\n")
    else:
        print("⚠️  GSD non disponibile — area_m2 non calcolata\n")

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

    # 4. Setup geo
    affine, geo_transformer = get_geo_tools(ORIGINAL_MOSAIC_PATH)
    if not affine:
        print("⚠️  GeoTIFF non trovato — KML/KMZ/CSV/GeoJSON e mosaico saltati.\n")
    raw_detections = []

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

        # Visualizzazione per patch
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

        # Geolocalizzazione
        if affine and geo_transformer:
            masks    = instances.pred_masks.numpy()
            boxes    = instances.pred_boxes.tensor.numpy()
            scores   = instances.scores.numpy()
            classes  = instances.pred_classes.numpy() if instances.has("pred_classes") else np.zeros(num_panels, dtype=int)

            for k in range(num_panels):
                mask_u8     = (masks[k].astype("uint8") * 255)
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

                area_m2 = calcola_area_m2(masks[k], gsd_m)

                raw_detections.append({
                    'lat':      lat,
                    'lon':      lon,
                    'gx':       gx,
                    'gy':       gy,
                    'score':    float(scores[k]),
                    'class_id': int(classes[k]),
                    'area_m2':  area_m2,
                    'contour':  global_contour
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
    final_panels = [p for p in final_panels if p['score'] >= SCORE_MIN_EXPORT]
    print(f"  Pannelli unici finali: {len(final_panels)}")

    # 7. KML + KMZ
    kml   = simplekml.Kml()
    style = simplekml.Style()
    style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    style.iconstyle.color     = 'ff0000ff'
    style.iconstyle.scale     = 0.5

    for p in final_panels:
        classe_nome     = CLASS_NAMES.get(p.get('class_id', 0), "PV_Module")
        pnt             = kml.newpoint(name="", coords=[(p['lon'], p['lat'])])
        pnt.description = (
            f"Classe: {classe_nome}\n"
            f"Score: {p['score']:.2f}\n"
            f"Area: {p.get('area_m2', 'N/A')} m²"
        )
        pnt.style = style

    kml.save(OUTPUT_KML)
    print(f"✅ KML salvato: {OUTPUT_KML}")
    esporta_kmz(OUTPUT_KML, OUTPUT_KMZ)

    # 8. CSV
    esporta_csv(final_panels, OUTPUT_CSV)

    # 9. GeoJSON
    esporta_geojson(final_panels, OUTPUT_GEOJSON)

    # 10. Mosaico annotato con contorni colorati per classe
    if os.path.exists(ORIGINAL_MOSAIC_PATH):
        try:
            mosaico = cv2.imread(ORIGINAL_MOSAIC_PATH)
            if mosaico is not None:
                for p in final_panels:
                    color = CLASS_COLORS.get(p.get('class_id', 0), (0, 200, 0))
                    if p.get('contour') is not None:
                        cv2.drawContours(mosaico, [p['contour']], -1, color, 2)
                        cx = int(p['gx'])
                        cy = int(p['gy'])
                        cv2.putText(
                            mosaico,
                            f"{p['score']:.2f}",
                            (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, color, 1, cv2.LINE_AA
                        )
                    else:
                        cv2.circle(
                            mosaico,
                            (int(p['gx']), int(p['gy'])),
                            DOT_RADIUS_MOSAIC, color, -1
                        )
                cv2.imwrite(OUTPUT_MOSAIC_MARKED, mosaico)
                print(f"✅ Mosaico salvato: {OUTPUT_MOSAIC_MARKED}")
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