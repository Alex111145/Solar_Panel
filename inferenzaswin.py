#!/usr/bin/env python
"""
inferenza_swinl.py - Inferenza MaskDINO Solar Panels con backbone Swin-L
=========================================================================
Aggiornamenti rispetto alla versione precedente:
  - YAML config aggiornato a Swin-L
  - OUTPUT_DIR punta a output_solar_swinl
  - get_best_weights cerca model_best.pth correttamente
  - setup_cfg include backbone Swin-L
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
import simplekml
import random
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
# ⚙️  CONFIGURAZIONE PARAMETRI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Modello ───────────────────────────────────────────────────────────────────
OUTPUT_DIR  = os.path.join(BASE_DIR, "output_solar_swinl")       # cartella training Swin-L
YAML_CONFIG = os.path.join(
    BASE_DIR, "configs", "coco", "instance-segmentation",
    "swin", "maskdino_swinl_bs16_50ep.yaml"
)
NUM_CLASSES = 1   # solar_panel

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_FOLDER_NAME = "solar_datasets_coco_segm_nuovo"
VAL_IMGS            = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "test")
ORIGINAL_MOSAIC_PATH = os.path.join(BASE_DIR, "ortomosaico.tif")

# ── Output ────────────────────────────────────────────────────────────────────
VIS_OUTPUT_DIR      = os.path.join(BASE_DIR, "inference_results")
OUTPUT_MOSAIC_MARKED = "Mosaico_Finale_Rilevato.jpg"
OUTPUT_KML          = "Mappa_Pannelli.kml"

# ── Soglie ────────────────────────────────────────────────────────────────────
SOGLIA_DETECTION = 0.37   # abbassa a 0.25 se perdi pannelli piccoli/tagliati
NMS_THRESH       = 0.50
MERGE_DIST_METERS = 1

# ── Grafica ───────────────────────────────────────────────────────────────────
DOT_RADIUS_MOSAIC = 8


# ==============================================================================
# 🛠️  FUNZIONI DI SUPPORTO
# ==============================================================================

def get_best_weights(output_dir):
    """
    Cerca il miglior checkpoint nell'output dir nell'ordine:
      1. model_best.pth  (salvato da BestCheckpointer quando segm/AP migliora)
      2. model_final.pth (ultimo checkpoint del training)
      3. Ultimo model_XXXXXXX.pth per data di creazione
    """
    best_model  = os.path.join(output_dir, "model_best.pth")
    final_model = os.path.join(output_dir, "model_final.pth")

    if os.path.exists(best_model):
        size_mb = os.path.getsize(best_model) / 1e6
        print(f"✅ Modello: model_best.pth ({size_mb:.0f} MB)  ← miglior AP sul val set")
        return best_model
    elif os.path.exists(final_model):
        size_mb = os.path.getsize(final_model) / 1e6
        print(f"⚠️  model_best.pth non trovato, uso model_final.pth ({size_mb:.0f} MB)")
        return final_model
    else:
        checkpoints = glob.glob(os.path.join(output_dir, "model_*.pth"))
        if checkpoints:
            latest = max(checkpoints, key=os.path.getctime)
            size_mb = os.path.getsize(latest) / 1e6
            print(f"⚠️  Uso ultimo checkpoint: {os.path.basename(latest)} ({size_mb:.0f} MB)")
            return latest
        else:
            print(f"❌ NESSUN modello .pth trovato in: {output_dir}")
            print(f"   Verifica che il training sia completato.")
            sys.exit(1)


def get_geo_tools(tif_path):
    """Carica il GeoTIFF e restituisce affine transform + coordinate transformer."""
    if not os.path.exists(tif_path):
        print(f"⚠️  GeoTIFF non trovato: {tif_path} — output KML e mosaico disabilitati")
        return None, None
    try:
        with rasterio.open(tif_path) as src:
            affine = src.transform
            crs    = src.crs
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        print(f"✅ GeoTIFF caricato: {os.path.basename(tif_path)}")
        return affine, transformer
    except Exception as e:
        print(f"⚠️  Errore caricamento GeoTIFF: {e}")
        return None, None


def filtra_punti_unici(detections, soglia_metri):
    """
    Fonde detection duplicate (stesso pannello rilevato in tile diverse).
    Ordina per score decrescente e tiene la detection migliore per ogni pannello.
    """
    unique_panels = []
    detections.sort(key=lambda x: x["score"], reverse=True)

    for p in detections:
        found = False
        for up in unique_panels:
            dist = math.sqrt(
                (p["lat"] - up["lat"]) ** 2 +
                (p["lon"] - up["lon"]) ** 2
            ) * 111000  # gradi → metri approssimativo
            if dist < soglia_metri:
                # Media pesata con le detection precedenti
                n = up["count"]
                up["lat"] = (up["lat"] * n + p["lat"]) / (n + 1)
                up["lon"] = (up["lon"] * n + p["lon"]) / (n + 1)
                up["gx"]  = (up["gx"]  * n + p["gx"])  / (n + 1)
                up["gy"]  = (up["gy"]  * n + p["gy"])  / (n + 1)
                up["count"] += 1
                found = True
                break
        if not found:
            unique_panels.append({**p, "count": 1})

    return unique_panels


def setup_cfg(weights_path):
    """
    Configura MaskDINO con backbone Swin-L.
    YAML_CONFIG deve puntare alla config Swin-L, non R50.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskdino_config(cfg)

    if not os.path.exists(YAML_CONFIG):
        print(f"❌ YAML non trovato: {YAML_CONFIG}")
        print("   Assicurati di aver copiato maskdino_swinl_bs16_50ep.yaml nella cartella swin/")
        sys.exit(1)

    cfg.merge_from_file(YAML_CONFIG)

    cfg.MODEL.WEIGHTS  = weights_path
    cfg.MODEL.DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

    # Numero classi: deve essere lo stesso usato nel training
    cfg.MODEL.ROI_HEADS.NUM_CLASSES    = NUM_CLASSES
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.MaskDINO.NUM_CLASSES     = NUM_CLASSES

    # Soglia score per inferenza
    # Nota: durante evaluate.py usiamo 0.05 per calcolare mAP
    # Qui usiamo 0.37 per avere risultati puliti visivamente
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SOGLIA_DETECTION

    print(f"✅ Config caricata: Swin-L | Device: {cfg.MODEL.DEVICE.upper()}")
    if cfg.MODEL.DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    return cfg


# ==============================================================================
# 🚀  MAIN
# ==============================================================================

def main():
    print("\n" + "=" * 60)
    print("🔍  INFERENZA SOLAR PANELS — MaskDINO Swin-L")
    print("=" * 60)

    # 1. Setup modello
    weights_path = get_best_weights(OUTPUT_DIR)
    cfg          = setup_cfg(weights_path)
    predictor    = DefaultPredictor(cfg)

    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

    # 2. Caricamento immagini
    all_files = sorted(
        glob.glob(os.path.join(VAL_IMGS, "*.jpg")) +
        glob.glob(os.path.join(VAL_IMGS, "*.JPG")) +
        glob.glob(os.path.join(VAL_IMGS, "*.png")) +
        glob.glob(os.path.join(VAL_IMGS, "*.PNG"))
    )

    if not all_files:
        print(f"❌ Nessuna immagine trovata in: {VAL_IMGS}")
        sys.exit(1)

    print(f"\n📂 Trovate {len(all_files)} immagini in: {VAL_IMGS}")

    # Selezione casuale
    scelta = input("👉 Quante immagini analizzare? (INVIO = tutte): ").strip()
    if scelta:
        try:
            limite    = min(int(scelta), len(all_files))
            all_files = random.sample(all_files, limite)
            print(f"🎲 Selezionate {len(all_files)} immagini casuali\n")
        except ValueError:
            print("⚠️  Input non valido, analizzo tutto il dataset\n")
    else:
        print(f"🚀 Analizzo tutte le {len(all_files)} immagini\n")

    # 3. Setup geo
    affine, geo_transformer = get_geo_tools(ORIGINAL_MOSAIC_PATH)
    raw_detections = []

    # 4. Loop inferenza
    total_panels = 0

    for i, path in enumerate(all_files):
        filename = os.path.basename(path)

        # Estrai offset dal nome tile (es: tile_col_1313_row_1213.jpg)
        match = re.search(r"tile_col_(\d+)_row_(\d+)", filename)
        off_x, off_y = (int(match.group(1)), int(match.group(2))) if match else (0, 0)

        img = cv2.imread(path)
        if img is None:
            print(f"⚠️  Impossibile leggere: {filename}")
            continue

        # Inferenza
        outputs   = predictor(img)
        instances = outputs["instances"].to("cpu")

        # NMS aggiuntivo
        if len(instances) > 0:
            keep      = nms(instances.pred_boxes.tensor, instances.scores, NMS_THRESH)
            instances = instances[keep]

        num_panels = len(instances)
        max_score  = instances.scores.max().item() if num_panels > 0 else 0.0
        total_panels += num_panels

        print(f"📸 [{i+1:>4}/{len(all_files)}] {filename:<45} | "
              f"Pannelli: {num_panels:>3} | Conf max: {max_score:.0%}")

        if num_panels == 0:
            continue

        # ── Visualizzazione ───────────────────────────────────────────────────
        try:
            meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        except Exception:
            meta = MetadataCatalog.get("__unused__")

        v = Visualizer(img[:, :, ::-1], meta, scale=1.0)
        v.overlay_instances(masks=instances.pred_masks)

        for k in range(num_panels):
            score_pct = f"{int(instances.scores[k].item() * 100)}%"
            v.draw_text(score_pct, instances.pred_boxes.tensor[k][:2].tolist())

        out = v.get_output()
        cv2.imwrite(
            os.path.join(VIS_OUTPUT_DIR, f"res_{filename}"),
            out.get_image()[:, :, ::-1]
        )

        # ── Geo-referenziazione ───────────────────────────────────────────────
        if affine is None or geo_transformer is None:
            continue

        masks  = instances.pred_masks.numpy()
        boxes  = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()

        for k in range(num_panels):
            mask_uint8 = (masks[k].astype("uint8") * 255)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            global_contour = None
            if len(contours) > 0:
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

            gx, gy         = off_x + cx_local, off_y + cy_local
            proj_x, proj_y = affine * (gx, gy)
            lon, lat        = geo_transformer.transform(proj_x, proj_y)

            raw_detections.append({
                "lat":     lat,
                "lon":     lon,
                "gx":      gx,
                "gy":      gy,
                "score":   float(scores[k]),
                "contour": global_contour,
            })

    # 5. Riepilogo detection
    print(f"\n{'=' * 60}")
    print(f"📊 RIEPILOGO: {total_panels} pannelli rilevati in {len(all_files)} immagini")
    print(f"{'=' * 60}")

    if not raw_detections:
        print("❌ Nessun pannello geo-referenziato (GeoTIFF mancante o nessuna detection).")
        print(f"✅ Immagini visualizzate salvate in: {VIS_OUTPUT_DIR}")
        sys.exit(0)

    # 6. Fusione duplicati
    print(f"\n🔄 Fusione duplicati (soglia: {MERGE_DIST_METERS} m)...")
    final_panels = filtra_punti_unici(raw_detections, MERGE_DIST_METERS)
    print(f"✅ Pannelli unici dopo fusione: {len(final_panels)}")

    # 7. KML
    kml   = simplekml.Kml()
    style = simplekml.Style()
    style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
    style.iconstyle.color     = "ff0000ff"   # rosso (formato AABBGGRR)
    style.iconstyle.scale     = 0.5

    for p in final_panels:
        pnt             = kml.newpoint(name="", coords=[(p["lon"], p["lat"])])
        pnt.description = f"Score: {p['score']:.2f} | Rilevamenti: {p['count']}"
        pnt.style       = style

    kml.save(OUTPUT_KML)
    print(f"✅ KML salvato: {OUTPUT_KML}  ({len(final_panels)} punti)")

    # 8. Mosaico annotato
    if os.path.exists(ORIGINAL_MOSAIC_PATH):
        try:
            mosaico = cv2.imread(ORIGINAL_MOSAIC_PATH)
            if mosaico is not None:
                for p in final_panels:
                    if p.get("contour") is not None:
                        cv2.drawContours(mosaico, [p["contour"]], -1, (0, 0, 255), 2)
                    else:
                        cv2.circle(
                            mosaico,
                            (int(p["gx"]), int(p["gy"])),
                            DOT_RADIUS_MOSAIC,
                            (0, 0, 255), -1
                        )
                cv2.imwrite(OUTPUT_MOSAIC_MARKED, mosaico)
                print(f"✅ Mosaico salvato: {OUTPUT_MOSAIC_MARKED}")
        except Exception as e:
            print(f"⚠️  Errore salvataggio mosaico: {e}")

    print(f"\n🎯 COMPLETATO — immagini in: {VIS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
