import cv2
import os
import glob
import random
import shutil
import numpy as np

# ==============================================================================
# ⚙️ CONFIGURAZIONE PERCORSI E PARAMETRI
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
TEST_DIR = os.path.join(BASE_DIR, "datasets", "solar-dataset-cocosegm", "test")

PERCENTUALE_TEST = 0.05 
TILE_SIZE = 800
OVERLAP = 0.80
IGNORE_EMPTY_THRESHOLD = 0.2

# --- Variabili globali per la selezione punti ---
points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            # Disegna un cerchio sul punto cliccato
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(param, points[-2], points[-1], (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(param, points[3], points[0], (0, 255, 0), 2)
            cv2.imshow("SELEZIONE 4 PUNTI", param)

# ==============================================================================
# 🚀 CORE PIPELINE
# ==============================================================================

def main():
    global points
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
    clone = preview_img.copy()

    # --- FASE 1: Selezione 4 Punti ---
    print("\n1️⃣ Clicca su 4 punti per definire l'area (es. i 4 angoli di un campo).")
    print("   Premi 'r' per resettare o 'c' per confermare dopo i 4 punti.")
    
    window_name = "SELEZIONE 4 PUNTI"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event, preview_img)

    while True:
        cv2.imshow(window_name, preview_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"): # Reset
            points = []
            preview_img = clone.copy()
            cv2.setMouseCallback(window_name, click_event, preview_img)
        elif key == ord("c") and len(points) == 4: # Confirm
            break
        elif key == 27: # ESC
            return

    cv2.destroyAllWindows()

    # --- FASE 2: Preparazione Maschera e Coordinate ---
    # Riportiamo i punti alla scala originale
    pts_orig = np.array([(int(p[0]/scale), int(p[1]/scale)) for p in points], np.int32)
    
    # Calcoliamo il rettangolo contenitore (bounding box) dei 4 punti
    x_s, y_s, w_box, h_box = cv2.boundingRect(pts_orig)
    x_e, y_e = x_s + w_box, y_s + h_box

    # Creiamo una maschera per salvare solo ciò che è dentro il quadrilatero (opzionale)
    # Qui la usiamo per decidere se processare una patch
    mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_orig], 255)

    # --- FASE 3: Taglio Patch ---
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    step = int(TILE_SIZE * (1 - OVERLAP))
    saved_count = 0

    print(f"\n✂️ Inizio taglio patch nell'area poligonale...")
    
    for y in range(y_s, min(y_e, h_orig) - TILE_SIZE + 1, step):
        for x in range(x_s, min(x_e, w_orig) - TILE_SIZE + 1, step):
            
            # Verifichiamo se il centro della patch è dentro il poligono
            # Questo evita di salvare patch completamente esterne ai 4 punti
            center_x, center_y = x + TILE_SIZE // 2, y + TILE_SIZE // 2
            if mask[center_y, center_x] == 0:
                continue

            patch = full_img[y : y + TILE_SIZE, x : x + TILE_SIZE]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            
            if (cv2.countNonZero(gray) / (TILE_SIZE**2)) >= IGNORE_EMPTY_THRESHOLD:
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"tile_col_{x}_row_{y}.jpg"), patch)
                saved_count += 1

    print(f"\n🎉 COMPLETATO TAGLIO! Patch salvate: {saved_count}")

    # --- FASE 4: Estrazione del Test Set (5%) ---
    # (Il resto del codice rimane invariato)
    all_files = glob.glob(os.path.join(OUTPUT_DIR, "*.jpg"))
    if not all_files: return

    num_test = max(1, int(len(all_files) * PERCENTUALE_TEST))
    os.makedirs(TEST_DIR, exist_ok=True)
    test_files = random.sample(all_files, num_test)

    print(f"\n🚚 Spostamento {num_test} immagini nel Test Set...")
    for file_path in test_files:
        shutil.move(file_path, os.path.join(TEST_DIR, os.path.basename(file_path)))

    print(f"🎉 Operazione conclusa con successo.")

if __name__ == "__main__":
    main()
