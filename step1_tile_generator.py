import cv2
import os
import json
import platform

# ==============================================================================
# ⚙️ CONFIGURAZIONE PERCORSI (Assoluti e Universali)
# ==============================================================================

def get_base_dir():
    # 1. Rileva la cartella dove si trova fisicamente questo script
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Lista dei percorsi assoluti probabili
    possible_paths = [
        os.path.expanduser("~/Desktop/Uni/Triennale/Tesi/python/tagliomosaico"),
        os.path.expanduser("~/DinoMask"),
        r"C:\Users\dell\Desktop\Tesi\python\tagliomosaico"
    ]

    for p in possible_paths:
        if os.path.exists(p):
            return p
    return current_script_path

BASE_DIR = get_base_dir()
INPUT_IMAGE_PATH = os.path.join(BASE_DIR, "ortomosaico.tif")
OUTPUT_DIR = os.path.join(BASE_DIR, "my_thesis_data")
ANCHOR_FILE = os.path.join(BASE_DIR, "anchor_pixel_coords.json")

TILE_SIZE = 800
OVERLAP = 0.95
IGNORE_EMPTY_THRESHOLD = 0.2

# Variabile globale per memorizzare il punto cliccato
anchor_coords = None

def select_anchor_point(event, x, y, flags, param):
    global anchor_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        scale = param['scale']
        # Converte le coordinate della preview in coordinate reali del mosaico originale
        real_x = int(x / scale)
        real_y = int(y / scale)
        anchor_coords = (real_x, real_y)
        print(f"\n📍 ANCORA SELEZIONATA: Pixel X={real_x}, Y={real_y}")

# ==============================================================================
# 🚀 CORE PIPELINE
# ==============================================================================

def main():
    global anchor_coords
    print(f"📂 Cartella di lavoro: {BASE_DIR}")

    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"❌ ERRORE: File {INPUT_IMAGE_PATH} non trovato!")
        return

    # 1. Caricamento Immagine
    print(f"🔄 Caricamento mosaico da: {INPUT_IMAGE_PATH}...")
    full_img = cv2.imread(INPUT_IMAGE_PATH)
    if full_img is None:
        print("❌ Errore: Impossibile leggere il file TIF.")
        return

    h_orig, w_orig = full_img.shape[:2]
    
    # Anteprima al 15% per la selezione
    scale = 0.15 
    preview_dim = (int(w_orig * scale), int(h_orig * scale))
    preview_img = cv2.resize(full_img, preview_dim, interpolation=cv2.INTER_AREA)

    window_name = "CONFIGURAZIONE DATASET"
    cv2.namedWindow(window_name)

    # --- FASE 1: Selezione Area di Interesse (ROI) ---
    print("\n1️⃣ Disegna il rettangolo dell'area da tagliare e premi SPAZIO o INVIO...")
    roi = cv2.selectROI(window_name, preview_img, fromCenter=False, showCrosshair=True)

    if roi == (0, 0, 0, 0):
        print("🛑 Selezione annullata.")
        return

    # --- FASE 2: Selezione Punto Ancora con istruzioni sulla GUI ---
    print("\n2️⃣ Seleziona il punto di ancora sulla GUI...")
    cv2.setMouseCallback(window_name, select_anchor_point, param={'scale': scale})
    
    while anchor_coords is None:
        temp_view = preview_img.copy()
        
        # Testo di istruzione sulla GUI
        instruction_text = "CLICCA SUL PUNTO DI ANCORA (Punto GPS noto)"
        cv2.putText(temp_view, instruction_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow(window_name, temp_view)
        # ESC per uscire forzatamente
        if cv2.waitKey(1) & 0xFF == 27: 
            break

    # Feedback visivo dopo il click
    if anchor_coords:
        confirm_view = preview_img.copy()
        # Disegna un cerchio nel punto cliccato (scalato per la preview)
        cv2.circle(confirm_view, (int(anchor_coords[0]*scale), int(anchor_coords[1]*scale)), 7, (0, 255, 0), -1)
        cv2.putText(confirm_view, "PUNTO PRESO! Premi un tasto per avviare il taglio...", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window_name, confirm_view)
        cv2.waitKey(0) 

        # Salvataggio delle coordinate pixel in un file JSON per lo Step 2
        with open(ANCHOR_FILE, 'w') as f:
            json.dump({"px_x": anchor_coords[0], "px_y": anchor_coords[1]}, f)
        print(f"✅ Coordinate ancora salvate in: {ANCHOR_FILE}")

    cv2.destroyAllWindows()
    for i in range(1, 5): cv2.waitKey(1)

    # --- FASE 3: Generazione delle Patch ---
    x_s, y_s, w, h = [int(v / scale) for v in roi]
    x_e, y_e = x_s + w, y_s + h

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    step = int(TILE_SIZE * (1 - OVERLAP))
    saved_count = 0

    print(f"\n✂️ Inizio taglio patch in corso...")
    for y in range(y_s, min(y_e, h_orig) - TILE_SIZE + 1, step):
        for x in range(x_s, min(x_e, w_orig) - TILE_SIZE + 1, step):
            patch = full_img[y : y + TILE_SIZE, x : x + TILE_SIZE]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            
            # Salva solo se la patch non è vuota (sfondo nero)
            if (cv2.countNonZero(gray) / (TILE_SIZE**2)) >= IGNORE_EMPTY_THRESHOLD:
                filename = f"tile_col_{x}_row_{y}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, filename), patch)
                saved_count += 1
                if saved_count % 50 == 0:
                    print(f"   ...salvate {saved_count} patch")

    print("-" * 40)
    print(f"🎉 COMPLETATO! Patch salvate: {saved_count}")
    print(f"📂 Cartella output: {OUTPUT_DIR}")
    print("-" * 40)

if __name__ == "__main__":
    main()
