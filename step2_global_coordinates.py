import json
import re
import os

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
ANCHOR_FILE = os.path.join(BASE_DIR, "anchor_pixel_coords.json")

def get_anchoring_data():
    print("\n⚓ CONFIGURAZIONE ANCORAGGIO GPS")
    
    # Caricamento automatico Pixel dallo Step 1
    px_x, px_y = None, None
    if os.path.exists(ANCHOR_FILE):
        with open(ANCHOR_FILE, 'r') as f:
            coords = json.load(f)
            px_x, px_y = coords['px_x'], coords['px_y']
            print(f"✅ Coordinate PIXEL caricate automaticamente dallo Step 1: X={px_x}, Y={px_y}")
    else:
        print("⚠️ File coordinate non trovato. Inserimento manuale richiesto:")
        px_x = float(input("   Pixel X: ").replace(',', '.'))
        px_y = float(input("   Pixel Y: ").replace(',', '.'))

    # Inserimento manuale solo per il GPS
    gps_lat = float(input("\n📍 Inserisci LATITUDINE Google Maps: ").strip().replace(',', '.'))
    gps_lon = float(input("📍 Inserisci LONGITUDINE Google Maps: ").strip().replace(',', '.'))
    
    return {
        'anchor_pixel_x': px_x, 'anchor_pixel_y': px_y,
        'anchor_gps_lat': gps_lat, 'anchor_gps_lon': gps_lon,
        'status': 'calibrated'
    }

def convert_to_global():
    if not os.path.exists(INPUT_JSON):
        print(f"❌ File non trovato: {INPUT_JSON}"); return

    georef_data = get_anchoring_data()

    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    # Logica di conversione (Invariata)
    for ann in data['annotations']:
        img = next(i for i in data['images'] if i['id'] == ann['image_id'])
        match = re.search(r"tile_col_(\d+)_row_(\d+)", img['file_name'])
        if match:
            off_x, off_y = int(match.group(1)), int(match.group(2))
            ann['bbox'][0] += off_x
            ann['bbox'][1] += off_y
            if 'segmentation' in ann:
                ann['segmentation'] = [[v + (off_x if i%2==0 else off_y) for i,v in enumerate(p)] for p in ann['segmentation']]

    data['georeference'] = georef_data
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n✅ MASTER JSON CREATO: {OUTPUT_JSON}")

if __name__ == "__main__":
    convert_to_global()
