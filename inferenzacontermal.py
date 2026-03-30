#!/usr/bin/env python
"""
inference.py - Inferenza MaskDINO Solar Panels (v4 - FINAL)
============================================================
Coerente con il training v4:
  - Risoluzione test 1600px (identica al training)
  - Soglia detection 0.37
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
import requests

# DJI Thermal Analysis Tool 3 — SDK Python (operante in background)
try:
    from dji_thermal_sdk.dji_sdk import (
        dirp_create_from_rjpeg,
        dirp_set_measurement_params,
        dirp_destroy,
    )
    from dji_thermal_sdk.utility import rjpeg_to_temperature_array
    DJI_SDK_OK = True
except ImportError:
    DJI_SDK_OK = False

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
# ⚙️ CONFIGURAZIONE
# ==============================================================================
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "output_solar_swinLb")         # ← RIGA 1: cartella output training
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

YAML_CONFIG = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation", "swin", "maskdino_swinl_bs16_50ep.yaml")  # ← RIGA 2: file YAML backbone
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
# 🌡️ CONFIGURAZIONE ANALISI TERMICA (DJI Thermal Analysis Tool 3)
# ==============================================================================
THERMAL_DIR    = os.path.join(BASE_DIR, "thermal")  # cartella file termici accoppiati (RJPEG/TIFF)
EMISSIVITA     = 0.95    # ε vetro temperato borosilicato/soda-lime moduli PV (fisso)
T_AMB_CELSIUS  = 25.0    # T_amb [°C] default — sostituita da input utente a runtime
T_STC          = 25.0    # temperatura STC riferimento IEC 60904 [°C]
DIST_OGGETTO_M = 50.0    # distanza di volo stimata [m]
UMIDITA_REL    = 70.0    # umidità relativa [%]

# Parametri per tecnologia cella — selezionati a runtime dall'utente
# γ: coeff. temperatura potenza massima [°C⁻¹] | η_nom: efficienza nominale STC
TECNOLOGIE_PANNELLO = {
    "1": {"nome": "Monocristallino (mono-Si)", "gamma": -0.0045, "eta_nom": 0.20},
    "2": {"nome": "Policristallino (poly-Si)", "gamma": -0.0040, "eta_nom": 0.17},
}

# ==============================================================================
# ☀️ CONFIGURAZIONE PVGIS
# ==============================================================================
PVGIS_BASE_URL       = "https://re.jrc.ec.europa.eu/api/v5_2"
PVGIS_TIMEOUT        = 30    # secondi

POTENZA_PANNELLO_KWP = 0.40  # kWp per pannello (default 400 W)
PERDITE_SISTEMA      = 14    # % perdite (cablaggio, inverter, sporco)
INCLINAZIONE_DEFAULT = 35    # ° tilt (0 = orizzontale)
ORIENTAMENTO_DEFAULT = 0     # ° azimuth (0 = Sud, -90 = Est, 90 = Ovest)

CO2_COEFF_KG_KWH     = 0.233 # kgCO₂ evitata per kWh (media EU grid mix)

MESI_IT = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
           "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]

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


def trova_file_termico(img_path, thermal_dir):
    """
    Cerca il file termico (RJPEG o TIFF 32-bit) accoppiato all'immagine RGB.
    Stessa naming convention (tile_col_X_row_Y), directory separata.
    """
    if not os.path.isdir(thermal_dir):
        return None
    basename = os.path.splitext(os.path.basename(img_path))[0]
    for ext in [".JPG", ".jpg", ".tif", ".TIF", ".tiff", ".TIFF"]:
        candidate = os.path.join(thermal_dir, basename + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def carica_array_termico_dji(thermal_path, t_amb_c, epsilon=EMISSIVITA,
                              dist_m=DIST_OGGETTO_M, humidity=UMIDITA_REL):
    """
    Carica l'array di temperatura [°C] tramite DJI Thermal Analysis Tool 3.

    - RJPEG + DJI SDK: inietta ε e T_amb nel motore DJI → restituisce T_reale
    - TIFF 32-bit (fallback): legge T_apparente grezza dal raster radiometrico

    Returns: (np.ndarray float32 [°C], corrected: bool)
    """
    ext = os.path.splitext(thermal_path)[1].lower()

    if DJI_SDK_OK and ext in ['.jpg', '.jpeg']:
        try:
            with open(thermal_path, 'rb') as f:
                raw = f.read()
            # Iniezione programmatica parametri → correzione Stefan-Boltzmann integrata nel motore DJI
            arr = rjpeg_to_temperature_array(
                raw,
                emissivity=epsilon,
                reflected_apparent_temperature=t_amb_c,
                distance=dist_m,
                humidity=humidity,
            )
            return arr.astype(np.float32), True   # T_reale [°C] già corretta
        except Exception as e:
            print(f"  ⚠️  DJI SDK: {e} — lettura TIFF diretta")

    # Fallback: TIFF 32-bit (T_apparente grezza)
    try:
        with rasterio.open(thermal_path) as src:
            arr = src.read(1).astype(np.float32)
        return arr, False   # T_apparente [°C], correzione da applicare
    except Exception as e:
        print(f"  ⚠️  Errore file termico: {e}")
        return None, False


def correggi_emissivita(t_apparente_c, t_amb_c, epsilon=EMISSIVITA):
    """
    Legge di Stefan-Boltzmann — Eq. radiometrica (correzione emissività):

        T_reale = [ (T_app⁴ − (1−ε)·T_amb⁴) / ε ]^(1/4)   [K → °C]
    """
    T_app = t_apparente_c + 273.15
    T_amb = t_amb_c + 273.15
    radicando = (T_app ** 4 - (1.0 - epsilon) * T_amb ** 4) / epsilon
    if radicando <= 0:
        return t_apparente_c
    return round((radicando ** 0.25) - 273.15, 2)


def calcola_efficienza_termica(t_reale_c, eta_nom, gamma):
    """
    Modello lineare di degradazione termica (IEC 61853):

        η_reale = η_nom · [1 + γ · (T_reale − T_STC)]
    """
    return round(eta_nom * (1.0 + gamma * (t_reale_c - T_STC)), 6)


def estrai_dato_termico(thermal_array, mask, class_id, t_amb_c, epsilon, corrected,
                         eta_nom, gamma):
    """
    Applica la maschera poligonale MaskDINO sull'array termico DJI ed estrae
    T_apparente, T_reale e η_reale per il pannello rilevato.

    Logica statistica per classe:
    - PV_Module (0) / Degrado (2): media matematica dei pixel ROI (modulo integro)
    - Hotspot (1):                 valore massimo assoluto (picco termico del difetto)
    """
    # Allinea risoluzione maschera all'array termico se necessario
    if thermal_array.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (thermal_array.shape[1], thermal_array.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    pixel_vals = thermal_array[mask.astype(bool)]
    if len(pixel_vals) == 0:
        return None, None, None

    # Estrazione statistica in base alla classe
    if class_id == 1:       # Hotspot → picco termico massimo
        t_stat = float(np.max(pixel_vals))
    else:                   # PV_Module / Degrado → media matematica
        t_stat = float(np.mean(pixel_vals))

    if corrected:
        # Il motore DJI ha già applicato la correzione emissività → t_stat è T_reale
        t_reale     = round(t_stat, 2)
        t_apparente = None
    else:
        # TIFF grezzo → applica correzione Stefan-Boltzmann
        t_apparente = round(t_stat, 2)
        t_reale     = correggi_emissivita(t_stat, t_amb_c, epsilon)

    eta = calcola_efficienza_termica(t_reale, eta_nom, gamma)
    return t_apparente, t_reale, eta


def geocodifica_citta(citta: str):
    """
    Converte il nome città in (lat, lon) tramite Nominatim (OpenStreetMap).
    Restituisce (lat, lon, display_name) oppure (None, None, None).
    """
    try:
        url    = "https://nominatim.openstreetmap.org/search"
        params = {"q": citta, "format": "json", "limit": 1}
        headers = {"User-Agent": "SolarPanelDetector/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", citta)
        return None, None, None
    except Exception as e:
        print(f"  ⚠️  Geocodifica fallita: {e}")
        return None, None, None


def ottieni_dati_pvgis(lat: float, lon: float, peakpower_kwp: float = POTENZA_PANNELLO_KWP,
                        loss: float = PERDITE_SISTEMA, tilt: float = INCLINAZIONE_DEFAULT,
                        azimuth: float = ORIENTAMENTO_DEFAULT):
    """
    Chiama PVGIS API v5.2 — endpoint /PVcalc.
    Restituisce il JSON con produzione mensile/annuale per 1 kWp, oppure None.
    """
    try:
        resp = requests.get(
            f"{PVGIS_BASE_URL}/PVcalc",
            params={
                "lat": lat, "lon": lon,
                "peakpower": peakpower_kwp, "loss": loss,
                "angle": tilt, "aspect": azimuth,
                "pvtechchoice": "crystSi", "mountingplace": "free",
                "outputformat": "json",
            },
            timeout=PVGIS_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        print("  ⚠️  PVGIS: timeout connessione")
    except requests.exceptions.ConnectionError:
        print("  ⚠️  PVGIS: impossibile connettersi (verifica connessione internet)")
    except Exception as e:
        print(f"  ⚠️  PVGIS /PVcalc errore: {e}")
    return None


def ottieni_irradiazione_pvgis(lat: float, lon: float):
    """
    Chiama PVGIS API v5.2 — endpoint /MRcalc.
    Restituisce il JSON con irradiazione mensile orizzontale e su piano ottimale, oppure None.
    """
    try:
        resp = requests.get(
            f"{PVGIS_BASE_URL}/MRcalc",
            params={
                "lat": lat, "lon": lon,
                "horirrad": 1, "optrad": 1,
                "outputformat": "json",
            },
            timeout=PVGIS_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ⚠️  PVGIS /MRcalc errore: {e}")
    return None


def stampa_report_pvgis(dati_pv, dati_irr, num_pannelli: int, potenza_pannello_kwp: float,
                         tech_info: dict, citta: str):
    """
    Stampa il report completo: irradiazione mensile, produzione stimata e CO₂ evitata.
    """
    print(f"\n{'='*60}")
    print(f"  ☀️  ANALISI PVGIS — {citta.upper()}")
    print(f"{'='*60}")

    # --- Irradiazione mensile ---
    if dati_irr:
        try:
            mesi_irr = dati_irr["outputs"]["monthly"]["fixed"]
            print(f"\n  📊 Irradiazione mensile (kWh/m²)")
            print(f"  {'Mese':<6} {'Orizz. H(h)':<16} {'Ottimale H(opt)'}")
            for i, m in enumerate(mesi_irr):
                h_h   = m.get("H(h)_m",     m.get("Hh",    0))
                h_opt = m.get("H(i_opt)_m", m.get("Hiopt", 0))
                nome  = MESI_IT[i] if i < 12 else f"M{i+1}"
                print(f"  {nome:<6} {h_h:<16.1f} {h_opt:.1f}")
        except (KeyError, TypeError) as e:
            print(f"  ⚠️  Dati irradiazione non disponibili: {e}")

    # --- Produzione PV mensile ---
    if dati_pv:
        try:
            out_mensili  = dati_pv["outputs"]["monthly"]["fixed"]
            out_annuale  = dati_pv["outputs"]["totals"]["fixed"]

            e_ann_1kwp   = out_annuale.get("E_y", 0)            # kWh/anno per 1 kWp
            h_irr_ann    = out_annuale.get("H(i)_y", out_annuale.get("H(i)_m", 0))

            potenza_tot_kwp  = num_pannelli * potenza_pannello_kwp
            prod_tot_ann_kwh = e_ann_1kwp * potenza_tot_kwp

            print(f"\n  ⚡ Produzione stimata — {num_pannelli} pannelli × {potenza_pannello_kwp*1000:.0f} W")
            print(f"     Potenza totale impianto: {potenza_tot_kwp:.2f} kWp")
            print(f"  {'Mese':<6} {'E/mese (1kWp)':<18} {'E/mese impianto (kWh)'}")
            for i, m in enumerate(out_mensili):
                e_m   = m.get("E_m", 0)
                nome  = MESI_IT[i] if i < 12 else f"M{i+1}"
                print(f"  {nome:<6} {e_m:<18.1f} {e_m * potenza_tot_kwp:.1f}")

            print(f"  {'─'*56}")
            print(f"  📈 Produzione annua (1 kWp):    {e_ann_1kwp:.0f} kWh/anno")
            print(f"  📈 Produzione annua (impianto): {prod_tot_ann_kwh:.0f} kWh/anno")
            print(f"  🌞 Irradiazione piano inclinato:{h_irr_ann:.0f} kWh/m²/anno")
            print(f"  ⚙️  Tecnologia:    {tech_info['nome']}")
            print(f"  📐 Inclinazione:  {INCLINAZIONE_DEFAULT}° | Orientamento: "
                  f"{'Sud' if ORIENTAMENTO_DEFAULT == 0 else str(ORIENTAMENTO_DEFAULT) + '°'}")
            print(f"  🔧 Perdite sistema: {PERDITE_SISTEMA}%")

            co2_kg = prod_tot_ann_kwh * CO2_COEFF_KG_KWH
            print(f"\n  🌿 CO₂ evitata/anno: {co2_kg:.0f} kg  ({co2_kg/1000:.2f} ton)")
        except (KeyError, TypeError) as e:
            print(f"  ⚠️  Dati produzione non disponibili: {e}")

    print(f"{'='*60}\n")


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
            "latitudine":    round(p['lat'], 8),
            "longitudine":   round(p['lon'], 8),
            "classe":        CLASS_NAMES.get(p.get('class_id', 0), "PV_Module"),
            "score":         round(p['score'], 4),
            "area_m2":       p.get('area_m2', None),
            "pixel_x":       int(p['gx']),
            "pixel_y":       int(p['gy']),
            "t_apparente_c": p.get('t_apparente', None),
            "t_reale_c":     p.get('t_reale', None),
            "eta_reale":     p.get('eta_reale', None),
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

    # 3b. Temperatura ambiente al momento del volo (stazione meteo locale)
    t_amb_input = input(
        f"🌡️  Temperatura ambiente al volo [°C] (INVIO = {T_AMB_CELSIUS}°C): "
    ).strip()
    try:
        t_amb_corrente = float(t_amb_input) if t_amb_input else T_AMB_CELSIUS
    except ValueError:
        t_amb_corrente = T_AMB_CELSIUS

    # 3c. Tipologia tecnologia cella fotovoltaica
    print("🔲 Tipologia pannello fotovoltaico:")
    for k, v in TECNOLOGIE_PANNELLO.items():
        print(f"   {k} → {v['nome']:<30} γ = {v['gamma']} °C⁻¹ | η_nom = {v['eta_nom']:.0%}")
    tech_input = input("👉 Scelta (INVIO = 1 Monocristallino): ").strip()
    tech_info      = TECNOLOGIE_PANNELLO.get(tech_input, TECNOLOGIE_PANNELLO["1"])
    gamma_corrente = tech_info["gamma"]
    eta_nom_corrente = tech_info["eta_nom"]

    sdk_stato = "✅ DJI SDK attivo" if DJI_SDK_OK else "⚠️  fallback TIFF diretto"
    print(f"\n  Tecnologia : {tech_info['nome']}")
    print(f"  γ          : {gamma_corrente} °C⁻¹  |  η_nom : {eta_nom_corrente:.0%}")
    print(f"  T_amb      : {t_amb_corrente}°C  |  ε (vetro) : {EMISSIVITA}  |  {sdk_stato}\n")

    # 3d. Città per analisi PVGIS
    citta_input = input("🌍 Città di installazione impianto (INVIO = salta analisi PVGIS): ").strip()
    pvgis_lat = pvgis_lon = pvgis_citta_display = None
    potenza_pannello_kwp = POTENZA_PANNELLO_KWP
    if citta_input:
        pvgis_lat, pvgis_lon, pvgis_citta_display = geocodifica_citta(citta_input)
        if pvgis_lat:
            print(f"  📍 {pvgis_citta_display}")
            print(f"     Coordinate: {pvgis_lat:.4f}°N, {pvgis_lon:.4f}°E")
            pot_input = input(
                f"  ⚡ Potenza per pannello [kWp] (INVIO = {POTENZA_PANNELLO_KWP} kWp): "
            ).strip()
            if pot_input:
                try:
                    potenza_pannello_kwp = float(pot_input)
                except ValueError:
                    potenza_pannello_kwp = POTENZA_PANNELLO_KWP
            print(f"  ✅ Potenza pannello: {potenza_pannello_kwp} kWp\n")
        else:
            print(f"  ⚠️  Città non trovata — analisi PVGIS saltata.\n")

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

                # Calcolo area degradata
                area_m2 = calcola_area_m2(masks[k], gsd_m)

                # Analisi termica tramite DJI Thermal Analysis Tool 3
                t_apparente = t_reale = eta_reale = None
                thermal_path = trova_file_termico(path, THERMAL_DIR)
                if thermal_path:
                    t_array, corrected = carica_array_termico_dji(
                        thermal_path, t_amb_corrente, EMISSIVITA,
                        DIST_OGGETTO_M, UMIDITA_REL
                    )
                    if t_array is not None:
                        t_apparente, t_reale, eta_reale = estrai_dato_termico(
                            t_array, masks[k], int(classes[k]),
                            t_amb_corrente, EMISSIVITA, corrected,
                            eta_nom_corrente, gamma_corrente
                        )

                raw_detections.append({
                    'lat':         lat,
                    'lon':         lon,
                    'gx':          gx,
                    'gy':          gy,
                    'score':       float(scores[k]),
                    'class_id':    int(classes[k]),
                    'area_m2':     area_m2,
                    'contour':     global_contour,
                    't_apparente': t_apparente,
                    't_reale':     t_reale,
                    'eta_reale':   eta_reale,
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
        classe_nome = CLASS_NAMES.get(p.get('class_id', 0), "PV_Module")
        pnt             = kml.newpoint(name="", coords=[(p['lon'], p['lat'])])
        t_app_str = f"{p['t_apparente']}°C" if p.get('t_apparente') is not None else "N/A"
        t_rea_str = f"{p['t_reale']}°C"    if p.get('t_reale')     is not None else "N/A"
        eta_str   = f"{p['eta_reale']:.4f}" if p.get('eta_reale')   is not None else "N/A"
        pnt.description = (
            f"Classe: {classe_nome}\n"
            f"Score: {p['score']:.2f}\n"
            f"Area: {p.get('area_m2', 'N/A')} m²\n"
            f"T_apparente: {t_app_str}\n"
            f"T_reale: {t_rea_str}\n"
            f"η_reale: {eta_str}"
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
                        # Annotazione score
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

    # 11. Analisi PVGIS
    if pvgis_lat is not None and len(final_panels) > 0:
        print("\n🔄 Recupero dati solari da PVGIS (JRC)...")
        dati_pv  = ottieni_dati_pvgis(
            pvgis_lat, pvgis_lon,
            peakpower_kwp=potenza_pannello_kwp,
        )
        dati_irr = ottieni_irradiazione_pvgis(pvgis_lat, pvgis_lon)
        if dati_pv or dati_irr:
            stampa_report_pvgis(
                dati_pv, dati_irr,
                len(final_panels), potenza_pannello_kwp,
                tech_info, citta_input,
            )
        else:
            print("  ⚠️  Nessun dato PVGIS disponibile.\n")

    print("🎯 PROCEDURA COMPLETATA.\n")


if __name__ == "__main__":
    main()
