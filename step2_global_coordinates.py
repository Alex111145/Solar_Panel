import json
import re
import os
import pyperclip
import time
import webbrowser
import subprocess
import platform
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
ANCHOR_FILE = os.path.join(BASE_DIR, "anchor_pixel_coords.json")

# ==============================================================================
# 🖥️ CONTROLLO AMBIENTE E GUI (X11 / DISPLAY CHECK)
# ==============================================================================

def is_ssh_connection():
    """Rileva se la sessione corrente è via SSH."""
    return 'SSH_CONNECTION' in os.environ or 'SSH_CLIENT' in os.environ

def print_x11_tutorial():
    """Stampa le istruzioni per attivare X11 se manca il display su server."""
    print("\n" + "!"*60)
    print("⚠️  ERRORE: INTERFACCIA GRAFICA NON DISPONIBILE")
    print("!"*60)
    print("\nSei collegato via SSH ma il server non può aprire il browser sul tuo PC.")
    print("\nPROVA A RISOLVERE COSÌ:")
    print("0. SCARICA VcXsrv (se non lo hai):")
    print("   -> https://sourceforge.net/projects/vcxsrv/")
    print("\n1. Su WINDOWS: Assicurati che VcXsrv (XLaunch) sia ATTIVO.")
    print("   -> Controlla che 'Disable access control' sia spuntato in XLaunch.")
    print("\n2. Su VS CODE: Apri il file config SSH e aggiungi queste righe:")
    print("      ForwardX11 yes")
    print("      ForwardX11Trusted yes")
    print("\n3. RIAVVIA la connessione SSH (Reload Window in VS Code).")
    print("\n4. VERIFICA: Digita 'echo $DISPLAY' nel terminale.")
    print("   -> Se non esce nulla, il tunnel X11 non è ancora attivo.")
    print("\n" + "!"*60 + "\n")

def check_display_environment():
    """Controlla se l'ambiente supporta una GUI. Se no, ferma l'esecuzione."""
    if not is_ssh_connection():
        return True
    
    if 'DISPLAY' not in os.environ:
        print_x11_tutorial()
        sys.exit(1) # Si ferma immediatamente
    
    print(f"✅ Display remoto rilevato: {os.environ['DISPLAY']}")
    return True

def open_maps_compact(url):
    """Apre la mappa in una finestrella dedicata (Modalità App)"""
    width, height = 300, 500 # Dimensioni ultra-compatte richieste
    
    try:
        if platform.system() == "Windows":
            # Forza finestra piccola su Edge o Chrome
            subprocess.Popen(f'start msedge --app="{url}" --window-size={width},{height}', shell=True)
        elif platform.system() == "Darwin": # macOS
            subprocess.Popen(['open', '-n', '-a', 'Google Chrome', '--args', f'--app={url}', f'--window-size={width},{height}'])
        else: # Linux (X11)
            subprocess.Popen(['google-chrome', f'--app={url}', f'--window-size={width},{height}'])
    except Exception as e:
        print(f"⚠️ Errore durante l'apertura del browser: {e}")
        webbrowser.open(url)

# ==============================================================================
# 🚀 CORE PIPELINE
# ==============================================================================

def convert_to_global():
    # --- CONTROLLO X11 PREVENTIVO ---
    check_display_environment()

    print(f"📂 Cartella di lavoro: {BASE_DIR}")
    
    if not os.path.exists(INPUT_JSON):
        print(f"❌ ERRORE: File {INPUT_JSON} non trovato!")
        return

    # --- FASE 1: Caricamento Pixel ---
    if os.path.exists(ANCHOR_FILE):
        with open(ANCHOR_FILE, 'r') as f:
            coords = json.load(f)
            px_x, px_y = coords['px_x'], coords['px_y']
            print(f"✅ Coordinate PIXEL caricate: X={px_x}, Y={px_y}")
    else:
        print("⚠️ File coordinate non trovato!")
        px_x = float(input("   Inserisci Pixel X manualmente: ").replace(',', '.'))
        px_y = float(input("   Inserisci Pixel Y manualmente: ").replace(',', '.'))

    # --- FASE 2: Acquisizione GPS (Browser) ---
    open_maps_compact("https://www.google.it/maps")
    
    print("\n" + "="*40)
    print("📋 ISTRUZIONI:")
    print("1. Tasto DESTRO sul punto scelto nella mappa.")
    print("2. Clicca sulle COORDINATE per copiarle negli appunti.")
    print("="*40)
    print("⌛ In attesa delle coordinate GPS...")

    gps_lat, gps_lon = None, None
    try:
        while True:
            content = pyperclip.paste().strip()
            match = re.search(r"(-?\d+\.\d+),\s*(-?\d+\.\d+)", content)
            
            if match:
                gps_lat = float(match.group(1))
                gps_lon = float(match.group(2))
                print(f"\n🎯 COORDINATE GPS RILEVATE!")
                print(f"   Lat: {gps_lat} | Lon: {gps_lon}")
                break
            
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n🛑 Operazione annullata.")
        return

    # --- FASE 3: Trasformazione Master JSON ---
    print("\n🔄 Aggiornamento Master JSON in corso...")
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    for ann in data['annotations']:
        img = next((i for i in data['images'] if i['id'] == ann['image_id']), None)
        if not img: continue
        
        match = re.search(r"tile_col_(\d+)_row_(\d+)", img['file_name'])
        if match:
            off_x, off_y = int(match.group(1)), int(match.group(2))
            ann['bbox'][0] += off_x
            ann['bbox'][1] += off_y
            if 'segmentation' in ann:
                ann['segmentation'] = [[v + (off_x if i%2==0 else off_y) for i,v in enumerate(p)] for p in ann['segmentation']]

    # Metadati georef
    data['georeference'] = {
        'anchor_pixel_x': px_x, 'anchor_pixel_y': px_y,
        'anchor_gps_lat': gps_lat, 'anchor_gps_lon': gps_lon,
        'status': 'calibrated'
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=4)
    
    print("-" * 40)
    print(f"🎉 MASTER JSON CREATO: {OUTPUT_JSON}")
    print("-" * 40)

if __name__ == "__main__":
    convert_to_global()
