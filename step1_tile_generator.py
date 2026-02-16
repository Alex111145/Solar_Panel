import cv2
import os
import sys
import platform

# ==============================================================================
# ⚙️ CONFIGURAZIONE PERCORSI (Assoluti e Universali)
# ==============================================================================

def get_base_dir():
    # 1. Prova a rilevare la cartella dello script (Metodo più affidabile)
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Percorsi specifici utente espansi (per Mac e Lambda)
    mac_path = os.path.expanduser("~/Desktop/Uni/Triennale/Tesi/python/tagliomosaico")
    lambda_path = os.path.expanduser("~/DinoMask")
    win_path = r"C:\Users\dell\Desktop\Tesi\python\tagliomosaico"

    if os.path.exists(mac_path):
        return mac_path
    elif os.path.exists(lambda_path):
        return lambda_path
    elif os.path.exists(win_path):
        return win_path
    else:
        # Se non trova i path predefiniti, usa la cartella dove è salvato lo script
        return current_script_path

BASE_DIR = get_base_dir()
INPUT_IMAGE_PATH = os.path.join(BASE_DIR, "ortomosaico.tif")
OUTPUT_DIR = os.path.join(BASE_DIR, "my_thesis_data")

TILE_SIZE = 800
OVERLAP = 0.95
IGNORE_EMPTY_THRESHOLD = 0.2

# ==============================================================================
# 🖥️ FUNZIONI DI CONTROLLO AMBIENTE
# ==============================================================================

def is_ssh_connection():
    """Rileva se la sessione corrente è via SSH."""
    return 'SSH_CONNECTION' in os.environ or 'SSH_CLIENT' in os.environ

def print_x11_tutorial():
    """Stampa le istruzioni per attivare X11 se manca il display su server."""
    print("\n" + "!"*60)
    print("⚠️  ERRORE: INTERFACCIA GRAFICA NON DISPONIBILE")
    print("!"*60)
    print("\nSei collegato via SSH ma il server non può inviare finestre al tuo PC.")
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

def check_display_connection():
    """Verifica se la GUI può essere aperta."""
    if not is_ssh_connection():
        current_os = platform.system()
        print(f"💻 Esecuzione locale rilevata ({current_os}). Apertura GUI consentita.")
        return True
    
    if 'DISPLAY' not in os.environ:
        print_x11_tutorial()
        return False
    
    print(f"✅ Display remoto rilevato: {os.environ['DISPLAY']}")
    return True

# ==============================================================================
# 🚀 CORE PIPELINE
# ==============================================================================

def main():
    if not check_display_connection():
        return

    print(f"📂 Cartella di lavoro: {BASE_DIR}")

    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"❌ ERRORE: File {INPUT_IMAGE_PATH} non trovato!")
        return

    # 2. Caricamento Immagine
    print(f"🔄 Caricamento mosaico da: {INPUT_IMAGE_PATH}...")
    full_img = cv2.imread(INPUT_IMAGE_PATH)
    if full_img is None:
        print("❌ Errore: Impossibile leggere il file TIF. Verifica l'integrità del file.")
        return

    h_orig, w_orig = full_img.shape[:2]
    
    # Anteprima al 15%
    scale = 0.15 
    preview_dim = (int(w_orig * scale), int(h_orig * scale))
    preview_img = cv2.resize(full_img, preview_dim, interpolation=cv2.INTER_AREA)

    print("\n" + "="*50)
    print("🖱️  ISTRUZIONI SELEZIONE:")
    print("   1. Disegna il rettangolo sui pannelli col mouse.")
    print("   2. Premi 'SPAZIO' o 'INVIO' per confermare.")
    print("   3. Premi 'C' per annullare.")
    print("="*50)

    try:
        window_name = "SELEZIONE AREA PANNELLI"
        roi = cv2.selectROI(window_name, preview_img, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        for i in range(1, 5): cv2.waitKey(1)
            
    except Exception as e:
        print(f"\n❌ ERRORE APERTURA FINESTRA: {e}")
        return

    if roi == (0, 0, 0, 0):
        print("🛑 Selezione annullata.")
        return

    # 5. Ricalcolo coordinate reali
    x_s, y_s, w, h = [int(v / scale) for v in roi]
    x_e, y_e = x_s + w, y_s + h

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    step = int(TILE_SIZE * (1 - OVERLAP))
    saved_count = 0

    print(f"\n✂️  Inizio taglio patch in corso...")
    for y in range(y_s, min(y_e, h_orig) - TILE_SIZE + 1, step):
        for x in range(x_s, min(x_e, w_orig) - TILE_SIZE + 1, step):
            patch = full_img[y : y + TILE_SIZE, x : x + TILE_SIZE]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
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