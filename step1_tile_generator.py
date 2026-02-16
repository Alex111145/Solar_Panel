import cv2
import os
import sys
import platform
import json

# ==============================================================================
# ⚙️ CONFIGURAZIONE PERCORSI
# ==============================================================================

def get_base_dir():
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    mac_path = os.path.expanduser("~/Desktop/Uni/Triennale/Tesi/python/tagliomosaico")
    lambda_path = os.path.expanduser("~/DinoMask")
    win_path = r"C:\Users\dell\Desktop\Tesi\python\tagliomosaico"

    if os.path.exists(mac_path): return mac_path
    elif os.path.exists(lambda_path): return lambda_path
    elif os.path.exists(win_path): return win_path
    else: return current_script_path

BASE_DIR = get_base_dir()
INPUT_IMAGE_PATH = os.path.join(BASE_DIR, "ortomosaico.tif")
OUTPUT_DIR = os.path.join(BASE_DIR, "my_thesis_data")
ANCHOR_FILE = os.path.join(BASE_DIR, "anchor_pixel_coords.json")

TILE_SIZE = 800
OVERLAP = 0.95
IGNORE_EMPTY_THRESHOLD = 0.2

anchor_coords = None

def select_anchor_point_callback(event, x, y, flags, param):
    global anchor_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        scale = param['scale']
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

    print(f"🔄 Caricamento mosaico...")
    full_img = cv2.imread(INPUT_IMAGE_PATH)
    if full_img is None: return

    h_orig, w_orig = full_img.shape[:2]
    scale = 0.15 
    preview_img = cv2.resize(full_img, (int(w_orig * scale), int(h_orig * scale)))

    window_name = "CONFIGURAZIONE DATASET"
    cv2.namedWindow(window_name)

    # --- FASE 1: Selezione Area (ROI) ---
    # Nota: cv2.selectROI gestisce la propria visualizzazione internamente
    print("\n1️⃣ Disegna il rettangolo dell'area da tagliare e premi SPAZIO o INVIO...")
    roi = cv2.selectROI(window_name, preview_img, fromCenter=False)

    if roi == (0, 0, 0, 0):
        print("🛑 Selezione annullata.")
        cv2.destroyAllWindows()
        return

    # --- FASE 2: Selezione Punto Ancora con scritte su GUI ---
    print("\n2️⃣ ORA SELEZIONA UN PUNTO COME ANCORA...")
    cv2.setMouseCallback(window_name, select_anchor_point_callback, param={'scale': scale})
    
    while anchor_coords is None:
        temp_view = preview_img.copy()
        # Aggiunta istruzioni grafiche sulla finestra
        cv2.putText(temp_view, "CLICCA UN PUNTO PER L'ANCORA (GPS)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
  
        cv2.imshow(window_name, temp_view)
        if cv2.waitKey(1) & 0xFF == 27: break 

    # Conferma visiva del punto selezionato
    if anchor_coords:
        with open(ANCHOR_FILE, 'w') as f:
            json.dump({"px_x": anchor_coords[0], "px_y": anchor_coords[1]}, f)
        
        # Feedback visivo finale prima di chiudere
        final_view = preview_img.copy()
        cv2.putText(final_view, "PUNTO SALVATO! Inizio taglio...", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(window_name, final_view)
        cv2.waitKey(1000) # Mostra per 1 secondo

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # --- FASE 3: Taglio Patch ---
    x_s, y_s, w, h = [int(v / scale) for v in roi]
    x_e, y_e = x_s + w, y_s + h

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    step = int(TILE_SIZE * (1 - OVERLAP))
    saved_count = 0

    print(f"\n✂️ Inizio taglio patch...")
    for y in range(y_s, min(y_e, h_orig) - TILE_SIZE + 1, step):
        for x in range(x_s, min(x_e, w_orig) - TILE_SIZE + 1, step):
            patch = full_img[y : y + TILE_SIZE, x : x + TILE_SIZE]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if (cv2.countNonZero(gray) / (TILE_SIZE**2)) >= IGNORE_EMPTY_THRESHOLD:
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"tile_col_{x}_row_{y}.jpg"), patch)
                saved_count += 1

    print(f"\n🎉 COMPLETATO! Patch salvate: {saved_count}")

if __name__ == "__main__":
    main()
