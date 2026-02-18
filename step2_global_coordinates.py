import json
import re
import os
import sys

# ==============================================================================
# ⚙️ CONFIGURAZIONE PERCORSI (Assoluti e Universali)
# ==============================================================================

def get_base_dir():
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.expanduser("~/Desktop/Uni/Triennale/Tesi/python/tagliomosaico"),
        os.path.expanduser("~/DinoMask"),
        r"C:\Users\dell\Desktop\Tesi\python\tagliomosaico"
    ]
    for p in possible_paths:
        if os.path.exists(p): return p
    return current_script_path

BASE_DIR = get_base_dir()
INPUT_JSON = os.path.join(BASE_DIR, "datasets", "solar_datasets", "train", "_annotations.coco.json")
OUTPUT_JSON = os.path.join(BASE_DIR, "master_global_anchored.json")
TFW_FILE = os.path.join(BASE_DIR, "ortomosaico.tfw")

# ==============================================================================
# 🧮 MOTORE DI TRASFORMAZIONE (Pixel -> GPS)
# ==============================================================================

def apply_affine_transform(px, py, tfw_params):
    """
    Converte coordinate Pixel (Globali) in Coordinate Geografiche (Lat/Lon o Metri)
    usando la matrice di trasformazione del file TFW.
    
    Formula Affine:
    X_geo = A*x + B*y + C
    Y_geo = D*x + E*y + F
    """
    A = tfw_params['pixel_size_x']
    D = tfw_params['rotation_y']
    B = tfw_params['rotation_x']
    E = tfw_params['pixel_size_y']
    C = tfw_params['origin_x']     # Easting / Longitude
    F = tfw_params['origin_y']     # Northing / Latitude

    geo_x = (A * px) + (B * py) + C
    geo_y = (D * px) + (E * py) + F
    
    return geo_y, geo_x  # Restituisce (Lat, Lon)

# ==============================================================================
# 🚀 CORE PIPELINE
# ==============================================================================

def convert_to_global():
    print(f"📂 Cartella di lavoro: {BASE_DIR}")

    # 1. Verifica File Input
    if not os.path.exists(INPUT_JSON):
        print(f"❌ ERRORE: File annotazioni {INPUT_JSON} non trovato!")
        return

    if not os.path.exists(TFW_FILE):
        print(f"❌ ERRORE CRITICO: File {TFW_FILE} non trovato!")
        print("   Questo script richiede il file .tfw per funzionare in automatico.")
        return

    # 2. Parsing del TFW
    print("\n📟 Lettura parametri georeferenziazione (.tfw)...")
    try:
        with open(TFW_FILE, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        if len(lines) < 6:
            raise ValueError("Il file .tfw ha meno di 6 righe.")

        tfw_data = {
            "pixel_size_x": float(lines[0]), # A
            "rotation_y":   float(lines[1]), # D
            "rotation_x":   float(lines[2]), # B
            "pixel_size_y": float(lines[3]), # E
            "origin_x":     float(lines[4]), # C
            "origin_y":     float(lines[5])  # F
        }
        
        print(f"✅ TFW Caricato correttamente.")
        print(f"   Scala X: {tfw_data['pixel_size_x']} | Scala Y: {tfw_data['pixel_size_y']}")
        print(f"   Origine: {tfw_data['origin_y']}, {tfw_data['origin_x']}")

    except Exception as e:
        print(f"❌ Errore lettura TFW: {e}")
        return

    # 3. Elaborazione Dati
    print("\n🔄 Elaborazione annotazioni e calcolo coordinate...")
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    processed_count = 0
    
    for ann in data['annotations']:
        img = next((i for i in data['images'] if i['id'] == ann['image_id']), None)
        if not img: continue
        
        # Estrazione offset dalla tile (es. tile_col_800_row_1600.jpg)
        match = re.search(r"tile_col_(\d+)_row_(\d+)", img['file_name'])
        if match:
            tile_off_x, tile_off_y = int(match.group(1)), int(match.group(2))
            
            # --- A. Trasformazione in PIXEL GLOBALI (Mosaico Intero) ---
            # Aggiorniamo la bbox per posizionarla nel "quadro grande"
            # bbox formato COCO: [x_min, y_min, width, height]
            local_x, local_y, w, h = ann['bbox']
            
            global_px_x = local_x + tile_off_x
            global_px_y = local_y + tile_off_y
            
            ann['bbox'][0] = global_px_x
            ann['bbox'][1] = global_px_y
            
            # Aggiornamento segmentazione se presente
            if 'segmentation' in ann:
                ann['segmentation'] = [[v + (tile_off_x if i%2==0 else tile_off_y) for i,v in enumerate(p)] for p in ann['segmentation']]

            # --- B. Trasformazione in GPS (Lat/Lon) ---
            # Calcoliamo il centroide del pannello in coordinate geografiche reali
            center_px_x = global_px_x + (w / 2)
            center_px_y = global_px_y + (h / 2)
            
            geo_lat, geo_lon = apply_affine_transform(center_px_x, center_px_y, tfw_data)
            
            # Salviamo i dati GPS direttamente nell'annotazione
            ann['gps_coordinates'] = {
                'latitude': geo_lat,
                'longitude': geo_lon
            }
            
            processed_count += 1

    # 4. Salvataggio e Metadati
    data['georeference'] = {
        'method': 'automatic_tfw_full_matrix',
        'source_file': 'ortomosaico.tfw',
        'affine_params': tfw_data,
        'status': 'georeferenced'
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=4)
    
    print("-" * 50)
    print(f"🎉 COMPLETATO! Elaborati {processed_count} pannelli.")
    print(f"💾 File salvato: {OUTPUT_JSON}")
    print("ℹ️  Ogni annotazione ora contiene il campo 'gps_coordinates'.")
    print("-" * 50)

if __name__ == "__main__":
    convert_to_global()
