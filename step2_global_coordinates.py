import json
import re
import os
import pyperclip
import time
import webbrowser
import subprocess
import platform

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

def open_maps_half_window(url):
    """Apre Google Maps in una finestrella compatta"""
    # Dimensioni ancora più ridotte: 600x500
    width, height = 300, 500
    try:
        if platform.system() == "Windows":
            # Usa Edge o Chrome in modalità app (senza barre superflue)
            subprocess.Popen(f'start msedge --app="{url}" --window-size={width},{height}', shell=True)
        elif platform.system() == "Darwin": # macOS
            subprocess.Popen(['open', '-n', '-a', 'Google Chrome', '--args', f'--app={url}', f'--window-size={width},{height}'])
        else: # Linux
            subprocess.Popen(['google-chrome', f'--app={url}', f'--window-size={width},{height}'])
    except Exception:
        webbrowser.open(url)

def get_anchoring_data():
    print("\n⚓ CONFIGURAZIONE ANCORAGGIO GPS (Smart Mode)")
    
    px_x, px_y = None, None
    if os.path.exists(ANCHOR_FILE):
        with open(ANCHOR_FILE, 'r') as f:
            coords = json.load(f)
            px_x, px_y = coords['px_x'], coords['px_y']
            print(f"✅ Coordinate PIXEL caricate: X={px_x}, Y={px_y}")
    else:
        print("⚠️ File coordinate non trovato. Inserimento manuale richiesto:")
        px_x = float(input("   Pixel X: ").replace(',', '.'))
        px_y = float(input("   Pixel Y: ").replace(',', '.'))

    # Apertura in finestra piccola
    open_maps_half_window("https://www.google.it/maps")
    
    print("\n--- ISTRUZIONI ---")
    print("1. Vai sul punto della mappa corrispondente all'anchor pixel.")
    print("2. Clicca con il TASTO DESTRO sul punto.")
    print("3. CLICCA SULLE COORDINATE (le prime in alto) per copiarle.")
    print("------------------")
    print("⌛ In attesa che tu copi le coordinate negli appunti...")

    gps_lat, gps_lon = None, None
    try:
        while True:
            content = pyperclip.paste().strip()
            match = re.search(r"(-?\d+\.\d+),\s*(-?\d+\.\d+)", content)
            
            if match:
                gps_lat = float(match.group(1))
                gps_lon = float(match.group(2))
                print(f"\n🎯 COORDINATE RILEVATE E INSERITE!")
                print(f"   Latitudine: {gps_lat}")
                print(f"   Longitudine: {gps_lon}")
                break
            
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n❌ Operazione annullata.")
        return None
    
    return {
        'anchor_pixel_x': px_x, 'anchor_pixel_y': px_y,
        'anchor_gps_lat': gps_lat, 'anchor_gps_lon': gps_lon,
        'status': 'calibrated'
    }

def convert_to_global():
    if not os.path.exists(INPUT_JSON):
        print(f"❌ File non trovato: {INPUT_JSON}"); return

    georef_data = get_anchoring_data()
    if not georef_data: return

    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

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