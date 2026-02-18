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

TILE_SIZE = 800
OVERLAP = 0.95
IGNORE_EMPTY_THRESHOLD = 0.2

# ==============================================================================
# 🚀 CORE PIPELINE
# ==============================================================================

def main():
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

    # Chiudiamo la finestra dopo la selezione ROI
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
