import json
import re
import os

# ==============================================================================
# ⚙️ CONFIGURAZIONE PERCORSI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Il file scaricato da Roboflow (Locali 0-800)
INPUT_JSON = os.path.join(BASE_DIR, "datasets", "solar_datasets", "train", "_annotations.coco.json")
# Il file finale che conterrà le coordinate dell'intero mosaico
OUTPUT_JSON = os.path.join(BASE_DIR, "master_global_anchored.json")

REGEX_PATTERN = r"tile_col_(\d+)_row_(\d+)"

def get_anchoring_data():
    """Chiede all'utente i dati per la calibrazione GPS millimetrica"""
    print("\n" + "="*50)
    print("⚓ CONFIGURAZIONE ANCORAGGIO MASTER JSON")
    print("="*50)
    
    try:
        # 1. Coordinate Pixel sul Mosaico Originale
        px_x = float(input("\n1️⃣ Inserisci coordinata PIXEL X dell'ancora sul mosaico: ").strip().replace(',', '.'))
        px_y = float(input("   Inserisci coordinata PIXEL Y dell'ancora sul mosaico: ").strip().replace(',', '.'))
        
        # 2. Coordinate GPS reali (Google Maps/Earth)
        gps_lat = float(input("\n2️⃣ Inserisci LATITUDINE Google Maps (es. 45.840736): ").strip().replace(',', '.'))
        gps_lon = float(input("   Inserisci LONGITUDINE Google Maps (es. 8.790503): ").strip().replace(',', '.'))
        
        return {
            'anchor_pixel_x': px_x,
            'anchor_pixel_y': px_y,
            'anchor_gps_lat': gps_lat,
            'anchor_gps_lon': gps_lon,
            'status': 'calibrated'
        }
    except ValueError:
        print("❌ Errore: Inserisci numeri validi!")
        exit()

def convert_to_global():
    if not os.path.exists(INPUT_JSON):
        print(f"❌ ERRORE: Non trovo il file {INPUT_JSON}. Controlla il percorso!")
        return

    # Richiesta dati di ancoraggio
    georef_data = get_anchoring_data()

    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    print(f"\n🔄 Elaborazione di {len(data['annotations'])} annotazioni...")

    # 1. Creazione mappa ID Immagine -> Offset (X, Y)
    # Leggiamo i nomi dei file 'tile_col_XXX_row_YYY.jpg' per sapere dove si trovava la patch
    image_id_to_offsets = {}
    for img in data['images']:
        fname = img['file_name']
        match = re.search(REGEX_PATTERN, fname)
        if match:
            off_x, off_y = int(match.group(1)), int(match.group(2))
            image_id_to_offsets[img['id']] = (off_x, off_y)
        else:
            print(f"⚠️ Warning: Nome file non riconosciuto: {fname}")

    # 2. Trasformazione delle coordinate da Locali a Globali
    count = 0
    for ann in data['annotations']:
        img_id = ann['image_id']
        
        if img_id in image_id_to_offsets:
            off_x, off_y = image_id_to_offsets[img_id]

            # Spostamento Bounding Box [x_min, y_min, width, height]
            ann['bbox'][0] += off_x  # Sposta X
            ann['bbox'][1] += off_y  # Sposta Y

            # Spostamento Segmentazione (Poligoni)
            if 'segmentation' in ann:
                new_segmentation = []
                for polygon in ann['segmentation']:
                    shifted_polygon = []
                    for i in range(0, len(polygon), 2):
                        shifted_polygon.append(polygon[i] + off_x)     # X globale
                        shifted_polygon.append(polygon[i+1] + off_y)   # Y globale
                    new_segmentation.append(shifted_polygon)
                ann['segmentation'] = new_segmentation
            
            count += 1

    # 3. Inserimento Metadati Geografici nel JSON
    # Questo rende il file un "Master JSON" completo con i dati di calibrazione
    data['georeference'] = georef_data

    # 4. Salvataggio del Master JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=4)

    print("-" * 50)
    print(f"✅ MASTER JSON CREATO CON SUCCESSO!")
    print(f"📊 Annotazioni globalizzate: {count}")
    print(f"📍 Dati di ancoraggio salvati internamente.")
    print(f"💾 File generato: {OUTPUT_JSON}")
    print("-" * 50)

if __name__ == "__main__":
    convert_to_global()