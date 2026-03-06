"""
=============================================================================
🚀 PIPELINE COMPLETA PER NUOVO SITO (SENZA ANNOTAZIONI)
=============================================================================
Questo script fa tutto il lavoro per un nuovo impianto fotovoltaico:

1. Taglia l'ortomosaico in tile (come taglio_mosaico.py)
2. Esegue l'inferenza MaskDINO su ogni tile (al posto dell'annotazione manuale)
3. Salva le predizioni in formato COCO JSON
4. Applica la trasformazione TFW per ottenere coordinate GPS

INPUT RICHIESTI:
- ortomosaico_nuovo.tif  (il mosaico del nuovo sito)
- ortomosaico_nuovo.tfw  (il file di georeferenziazione del nuovo sito)
- model_final.pth        (i pesi del modello addestrato sul sito vecchio)

OUTPUT:
- predizioni_gps.json    (ogni pannello trovato con le sue coordinate GPS)
=============================================================================
"""

import os
import sys
import json
import re
import glob
import cv2
import numpy as np

# ==============================================================================
# ⚙️ CONFIGURAZIONE - MODIFICA QUESTI PERCORSI PER IL TUO SETUP
# ==============================================================================

# --- INPUT ---
# Percorso dell'ortomosaico del NUOVO sito (quello del cliente)
ORTOMOSAICO_PATH = "ortomosaico_nuovo.tif"

# Percorso del file TFW del NUOVO sito (DEVE essere del nuovo mosaico!)
TFW_PATH = "ortomosaico_nuovo.tfw"

# Percorso del modello addestrato (quello che hai trainato sul sito vecchio)
MODEL_WEIGHTS = "output_tesi/model_final.pth"

# Percorso del file di configurazione di MaskDINO
CONFIG_FILE = "tesi_config.yaml"

# --- PARAMETRI TAGLIO ---
TILE_SIZE = 800
OVERLAP = 0.70  # Stesso overlap usato in training!

# --- PARAMETRI INFERENZA ---
CONFIDENCE_THRESHOLD = 0.5  # Soglia minima di confidenza (0.0 - 1.0)

# --- OUTPUT ---
OUTPUT_TILES_DIR = "tiles_nuovo_sito"
OUTPUT_JSON = "predizioni_gps.json"


# ==============================================================================
# 🔧 FASE 0: Import Detectron2 / MaskDINO
# ==============================================================================

def setup_detectron2():
    """
    Importa e configura Detectron2 con MaskDINO.
    Questa funzione viene chiamata solo se Detectron2 è installato.
    """
    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.data import MetadataCatalog
        # Importa il modulo MaskDINO (deve essere nel PYTHONPATH)
        # Se hai la cartella maskdino/ nella root del progetto, dovrebbe funzionare
        try:
            from maskdino import add_maskdino_config
        except ImportError:
            print("⚠️  Impossibile importare maskdino. Assicurati che la cartella maskdino/ sia accessibile.")
            print("   Prova: export PYTHONPATH=$PYTHONPATH:/percorso/del/progetto")
            return None

        cfg = get_cfg()
        add_maskdino_config(cfg)
        cfg.merge_from_file(CONFIG_FILE)
        
        cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
        # Se sei su Mac/CPU:
        if not cfg.MODEL.DEVICE:
            cfg.MODEL.DEVICE = "cpu"
        
        predictor = DefaultPredictor(cfg)
        print("✅ Modello MaskDINO caricato con successo!")
        return predictor
        
    except ImportError as e:
        print(f"❌ Detectron2 non trovato: {e}")
        print("   Installa con: python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'")
        return None


# ==============================================================================
# ✂️ FASE 1: TAGLIO ORTOMOSAICO IN TILE
# ==============================================================================

def taglio_tile(image_path, output_dir, tile_size=800, overlap=0.70):
    """
    Taglia l'ortomosaico in tile con la stessa logica dello script originale.
    I nomi dei file seguono il formato: tile_col_{X}_row_{Y}.jpg
    (fondamentale per la georeferenziazione!)
    """
    print(f"\n✂️ FASE 1: Taglio ortomosaico in tile...")
    print(f"   File: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ ERRORE: Ortomosaico {image_path} non trovato!")
        return 0

    full_img = cv2.imread(image_path)
    if full_img is None:
        print(f"❌ ERRORE: Impossibile leggere {image_path}")
        return 0

    h_orig, w_orig = full_img.shape[:2]
    print(f"   Dimensioni mosaico: {w_orig} x {h_orig} px")

    os.makedirs(output_dir, exist_ok=True)
    
    step = int(tile_size * (1 - overlap))
    saved_count = 0
    ignore_threshold = 0.2  # Ignora tile quasi vuote

    for y in range(0, h_orig - tile_size + 1, step):
        for x in range(0, w_orig - tile_size + 1, step):
            patch = full_img[y : y + tile_size, x : x + tile_size]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            
            # Salta tile quasi completamente nere/vuote
            if (cv2.countNonZero(gray) / (tile_size ** 2)) >= ignore_threshold:
                # IMPORTANTE: il nome segue lo STESSO formato dello script di training!
                filename = f"tile_col_{x}_row_{y}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), patch)
                saved_count += 1

    print(f"✅ Taglio completato! {saved_count} tile salvate in: {output_dir}")
    return saved_count


# ==============================================================================
# 🤖 FASE 2: INFERENZA SU TUTTE LE TILE
# ==============================================================================

def inferenza_su_tile(predictor, tiles_dir, confidence_threshold=0.5):
    """
    Esegue l'inferenza MaskDINO su ogni tile e raccoglie le predizioni
    in formato COCO JSON (lo stesso formato delle annotazioni manuali).
    
    Questo è il CUORE dello script: sostituisce l'annotazione manuale.
    """
    print(f"\n🤖 FASE 2: Inferenza MaskDINO su tutte le tile...")
    
    tile_files = sorted(glob.glob(os.path.join(tiles_dir, "tile_col_*_row_*.jpg")))
    
    if not tile_files:
        print(f"❌ Nessuna tile trovata in {tiles_dir}")
        return None

    print(f"   Tile da processare: {len(tile_files)}")

    # Struttura COCO JSON (identica a quella delle annotazioni manuali)
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "solar_panel"}]  # La tua categoria
    }

    annotation_id = 1
    
    for img_idx, tile_path in enumerate(tile_files):
        filename = os.path.basename(tile_path)
        img = cv2.imread(tile_path)
        
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Aggiungi l'immagine al JSON (come farebbe COCO)
        coco_output["images"].append({
            "id": img_idx + 1,
            "file_name": filename,
            "width": w,
            "height": h
        })

        # --- INFERENZA ---
        outputs = predictor(img)
        instances = outputs["instances"]
        
        # Filtra per confidenza
        if len(instances) > 0:
            scores = instances.scores.cpu().numpy()
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            
            # Se ci sono maschere di segmentazione
            has_masks = instances.has("pred_masks")
            if has_masks:
                masks = instances.pred_masks.cpu().numpy()

            for i in range(len(instances)):
                if scores[i] < confidence_threshold:
                    continue
                
                # Bbox in formato COCO: [x_min, y_min, width, height]
                x1, y1, x2, y2 = boxes[i]
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                
                ann = {
                    "id": annotation_id,
                    "image_id": img_idx + 1,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(bbox_w), float(bbox_h)],
                    "score": float(scores[i]),
                    "area": float(bbox_w * bbox_h),
                    "iscrowd": 0
                }
                
                # Aggiungi segmentazione se disponibile (da MaskDINO)
                if has_masks:
                    # Converti la maschera binaria in poligono COCO
                    mask = masks[i].astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    segmentation = []
                    for contour in contours:
                        if len(contour) >= 3:  # Minimo 3 punti per un poligono
                            seg = contour.flatten().tolist()
                            segmentation.append(seg)
                    
                    if segmentation:
                        ann["segmentation"] = segmentation

                coco_output["annotations"].append(ann)
                annotation_id += 1

        # Progresso
        if (img_idx + 1) % 50 == 0 or img_idx == 0:
            print(f"   Processate {img_idx + 1}/{len(tile_files)} tile... "
                  f"({len(coco_output['annotations'])} pannelli trovati)")

    print(f"\n✅ Inferenza completata!")
    print(f"   Pannelli rilevati totali: {len(coco_output['annotations'])}")
    
    return coco_output


# ==============================================================================
# 🌍 FASE 3: GEOREFERENZIAZIONE (GPS)
# ==============================================================================

def leggi_tfw(tfw_path):
    """Legge i parametri di trasformazione affine dal file .tfw"""
    print(f"\n🌍 FASE 3: Lettura parametri georeferenziazione...")
    
    if not os.path.exists(tfw_path):
        print(f"❌ ERRORE: File TFW {tfw_path} non trovato!")
        return None
    
    with open(tfw_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    if len(lines) < 6:
        print("❌ ERRORE: Il file TFW ha meno di 6 righe!")
        return None

    tfw_data = {
        "pixel_size_x": float(lines[0]),  # A
        "rotation_y":   float(lines[1]),  # D
        "rotation_x":   float(lines[2]),  # B
        "pixel_size_y": float(lines[3]),  # E
        "origin_x":     float(lines[4]),  # C (Easting/Lon)
        "origin_y":     float(lines[5])   # F (Northing/Lat)
    }
    
    print(f"✅ TFW caricato: Origine ({tfw_data['origin_y']}, {tfw_data['origin_x']})")
    return tfw_data


def apply_affine_transform(px, py, tfw):
    """Converte pixel globali -> coordinate geografiche"""
    geo_x = (tfw['pixel_size_x'] * px) + (tfw['rotation_x'] * py) + tfw['origin_x']
    geo_y = (tfw['rotation_y'] * px) + (tfw['pixel_size_y'] * py) + tfw['origin_y']
    return geo_y, geo_x  # (Lat, Lon)


def georeferenzia_predizioni(coco_data, tfw_data):
    """
    Prende le predizioni COCO e aggiunge le coordinate GPS.
    Logica IDENTICA a convert_to_global.py, ma lavora sulle predizioni
    del modello invece che sulle annotazioni manuali.
    """
    print(f"\n📍 Calcolo coordinate GPS per ogni pannello...")
    
    processed = 0
    
    for ann in coco_data['annotations']:
        # Trova l'immagine corrispondente
        img = next((i for i in coco_data['images'] if i['id'] == ann['image_id']), None)
        if not img:
            continue
        
        # Estrai offset dal nome tile (es. tile_col_800_row_1600.jpg)
        match = re.search(r"tile_col_(\d+)_row_(\d+)", img['file_name'])
        if not match:
            continue
        
        tile_off_x = int(match.group(1))
        tile_off_y = int(match.group(2))
        
        # Converti bbox locale -> pixel globali (nel mosaico intero)
        local_x, local_y, w, h = ann['bbox']
        global_px_x = local_x + tile_off_x
        global_px_y = local_y + tile_off_y
        
        # Aggiorna la bbox con le coordinate globali
        ann['bbox'][0] = global_px_x
        ann['bbox'][1] = global_px_y
        
        # Aggiorna segmentazione se presente
        if 'segmentation' in ann:
            ann['segmentation'] = [
                [v + (tile_off_x if i % 2 == 0 else tile_off_y) for i, v in enumerate(p)]
                for p in ann['segmentation']
            ]
        
        # Calcola il centroide in coordinate GPS
        center_px_x = global_px_x + (w / 2)
        center_px_y = global_px_y + (h / 2)
        
        geo_lat, geo_lon = apply_affine_transform(center_px_x, center_px_y, tfw_data)
        
        ann['gps_coordinates'] = {
            'latitude': geo_lat,
            'longitude': geo_lon
        }
        
        processed += 1

    # Aggiungi metadati
    coco_data['georeference'] = {
        'method': 'automatic_tfw_full_matrix',
        'source_file': os.path.basename(TFW_PATH),
        'affine_params': tfw_data,
        'status': 'georeferenced',
        'note': 'Predizioni automatiche (non annotazioni manuali)'
    }

    print(f"✅ Georeferenziati {processed} pannelli!")
    return coco_data


# ==============================================================================
# 🎯 MAIN: PIPELINE COMPLETA
# ==============================================================================

def main():
    print("=" * 60)
    print("🔆 PIPELINE NUOVO SITO - INFERENZA + GEOREFERENZIAZIONE")
    print("=" * 60)
    
    # ---- FASE 1: Taglio ----
    num_tiles = taglio_tile(
        ORTOMOSAICO_PATH, 
        OUTPUT_TILES_DIR, 
        tile_size=TILE_SIZE, 
        overlap=OVERLAP
    )
    
    if num_tiles == 0:
        print("❌ Nessuna tile generata. Controlla il percorso dell'ortomosaico.")
        return

    # ---- FASE 2: Inferenza ----
    predictor = setup_detectron2()
    if predictor is None:
        print("\n❌ Impossibile caricare il modello. Assicurati che Detectron2 e MaskDINO siano installati.")
        print("   Se sei su un altro PC senza GPU, copia i pesi model_final.pth e usa cfg.MODEL.DEVICE = 'cpu'")
        return

    coco_predictions = inferenza_su_tile(
        predictor, 
        OUTPUT_TILES_DIR, 
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    if coco_predictions is None or len(coco_predictions['annotations']) == 0:
        print("⚠️ Nessun pannello rilevato! Prova ad abbassare CONFIDENCE_THRESHOLD (es. 0.3)")
        return

    # ---- FASE 3: Georeferenziazione ----
    tfw_data = leggi_tfw(TFW_PATH)
    if tfw_data is None:
        return

    result = georeferenzia_predizioni(coco_predictions, tfw_data)

    # ---- Salvataggio ----
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(result, f, indent=4)

    print("\n" + "=" * 60)
    print(f"🎉 COMPLETATO!")
    print(f"💾 File salvato: {OUTPUT_JSON}")
    print(f"📊 Pannelli trovati: {len(result['annotations'])}")
    print(f"📍 Ogni pannello ha il campo 'gps_coordinates' con lat/lon")
    print("=" * 60)
    
    # Mostra esempio
    if result['annotations']:
        esempio = result['annotations'][0]
        print(f"\n📌 Esempio primo pannello:")
        print(f"   GPS: {esempio['gps_coordinates']['latitude']:.6f}, "
              f"{esempio['gps_coordinates']['longitude']:.6f}")
        print(f"   Confidenza: {esempio.get('score', 'N/A')}")


if __name__ == "__main__":
    main()
