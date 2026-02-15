import cv2
import os
import re

# ==============================================================================
# ⚙️ CONFIGURAZIONE PERCORSI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGE_PATH = os.path.join(BASE_DIR, "ortomosaico.tif")
OUTPUT_DIR = os.path.join(BASE_DIR, "my_thesis_data")

TILE_SIZE = 800
OVERLAP = 0.95
IGNORE_EMPTY_THRESHOLD = 0.2

def main():
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"❌ ERRORE: File {INPUT_IMAGE_PATH} non trovato!")
        return

    # 1. Caricamento Immagine
    print("🔄 Caricamento mosaico per anteprima...")
    full_img = cv2.imread(INPUT_IMAGE_PATH)
    if full_img is None:
        print("❌ Errore: Impossibile leggere il file TIF.")
        return

    h_orig, w_orig = full_img.shape[:2]

    # 2. Creazione anteprima per la GUI (Mac/Windows hanno schermi limitati)
    # Ridimensioniamo al 20% o meno per la selezione
    scale = 0.15 # 15% della dimensione reale
    preview_dim = (int(w_orig * scale), int(h_orig * scale))
    preview_img = cv2.resize(full_img, preview_dim, interpolation=cv2.INTER_AREA)

    # 3. Interfaccia GUI per Selezione ROI
    print("\n🖱️  ISTRUZIONI GUI:")
    print("   1. Trascina il mouse per disegnare il rettangolo sui PANNELLI.")
    print("   2. Puoi spostare o ridimensionare il rettangolo.")
    print("   3. Premi 'INVIO' o 'SPAZIO' per confermare e TAGLIARE.")
    print("   4. Premi 'C' per annullare.")

    # Apre la finestra di selezione
    roi = cv2.selectROI("SELEZIONE AREA PANNELLI - TESI", preview_img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("SELEZIONE AREA PANNELLI - TESI")

    if roi == (0, 0, 0, 0):
        print("🛑 Selezione annullata.")
        return

    # 4. Ricalcolo coordinate reali (Scaling inverso)
    # roi = (x_start, y_start, width, height) in pixel anteprima
    x_s, y_s, w, h = [int(v / scale) for v in roi]
    x_e, y_e = x_s + w, y_s + h

    print(f"\n✅ AREA SELEZIONATA (Coordinate Reali):")
    print(f"   X: {x_s} -> {x_e} | Y: {y_s} -> {y_e}")

    # 5. Avvio Taglio Patch
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    step = int(TILE_SIZE * (1 - OVERLAP))
    saved_count = 0

    print(f"\n✂️  Inizio taglio patch nella zona selezionata...")
    
    # Ciclo limitato alla ROI scelta dall'utente
    for y in range(y_s, min(y_e, h_orig) - TILE_SIZE + 1, step):
        for x in range(x_s, min(x_e, w_orig) - TILE_SIZE + 1, step):
            
            patch = full_img[y : y + TILE_SIZE, x : x + TILE_SIZE]
            
            # Controllo pixel neri
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if (cv2.countNonZero(gray) / (TILE_SIZE**2)) >= IGNORE_EMPTY_THRESHOLD:
                filename = f"tile_col_{x}_row_{y}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, filename), patch)
                saved_count += 1

    print("-" * 40)
    print(f"🎉 COMPLETATO!")
    print(f"💾 Patch salvate: {saved_count}")
    print(f"📍 Destinazione: {OUTPUT_DIR}")
    print("-" * 40)

if __name__ == "__main__":
    main()