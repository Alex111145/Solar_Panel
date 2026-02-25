import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch

# ==============================================================================
# ⚙️ CONFIGURAZIONE PERCORSI E NOMI FILE
# ==============================================================================
TIF_PATH = "ortomosaico.tif"
OUTPUT_TILING = "schema_tiling.png"
OUTPUT_AFFINE = "schema_affine.png"

# ==============================================================================
# 1. GENERAZIONE SCHEMA TILING (Proporzioni Reali Dinamiche)
# ==============================================================================
def genera_schema_tiling():
    print("⏳ Generazione Schema Tiling (Lettura dimensioni reali in corso)...")
    
    fig, ax_main = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('white')
    
    # 1. Lettura delle dimensioni ESATTE dal file TIF
    try:
        with rasterio.open(TIF_PATH) as src:
            img_w, img_h = src.width, src.height
            print(f"📊 Dimensioni Reali Mosaico: {img_w} x {img_h} pixel")
            
            # Carichiamo l'immagine rimpicciolita solo per non saturare la RAM
            # MA manteniamo l'EXTENT sui valori originali per il tracciamento
            scale_factor = 0.05  
            h_small = int(img_h * scale_factor)
            w_small = int(img_w * scale_factor)
            img = src.read(1, out_shape=(h_small, w_small)) 
            
            # extent=[left, right, bottom, top] -> L'asse Y parte da 0 in alto
            ax_main.imshow(img, cmap='gray', alpha=0.75, extent=[0, img_w, img_h, 0])
            
    except Exception as e:
        print(f"⚠️ Impossibile caricare il TIF ({e}). Uso coordinate di simulazione.")
        img_w, img_h = 20000, 15000  # Dimensioni tipiche di un volo drone
        ax_main.imshow(np.random.rand(10, 10), cmap='gray', alpha=0.2, extent=[0, img_w, img_h, 0])

    # 2. Disegno della griglia globale esatta (800x800)
    tile_w, tile_h = 800, 800
    
    # Se il mosaico è enorme, mostriamo le linee ogni 800 pixel ma molto sottili
    for x in range(0, img_w, tile_w):
        ax_main.axvline(x, color='#4169E1', linestyle='-', linewidth=0.3, alpha=0.5)
    for y in range(0, img_h, tile_h):
        ax_main.axhline(y, color='#4169E1', linestyle='-', linewidth=0.3, alpha=0.5)

    # 3. Calcolo automatico di una Patch centrale (allineata alla griglia di 800)
    target_x = (img_w // 2) - ((img_w // 2) % tile_w)
    target_y = (img_h // 2) - ((img_h // 2) % tile_h)
    
    rect_tile = patches.Rectangle((target_x, target_y), tile_w, tile_h, 
                                  linewidth=3, edgecolor='red', facecolor='yellow', alpha=0.6)
    ax_main.add_patch(rect_tile)
    
    # Etichetta dinamica che si sposta sopra al quadrato
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9)
    
    # Regoliamo l'altezza del testo in base alla grandezza dell'immagine
    text_offset_y = img_h * 0.03 
    ax_main.text(target_x, target_y - text_offset_y, f"Patch: tile_col_{target_x}_row_{target_y}", 
                 color='red', fontsize=12, fontweight='bold', bbox=bbox_props)

    # 4. Box dello Zoom (Fisso sulla destra)
    ax_zoom = fig.add_axes([0.65, 0.50, 0.30, 0.35]) 
    ax_zoom.set_title(f"Sistema di Riferimento Locale ({tile_w}x{tile_h})", fontsize=11, fontweight='bold')
    ax_zoom.set_xlim(0, tile_w)
    ax_zoom.set_ylim(tile_h, 0) # Invertito per matchare le coordinate immagine
    ax_zoom.set_facecolor('#f8f9fa')
    
    # Griglia locale ogni 100 pixel
    ax_zoom.set_xticks(np.arange(0, tile_w+1, 200))
    ax_zoom.set_yticks(np.arange(0, tile_h+1, 200))
    ax_zoom.grid(True, linestyle='--', alpha=0.7)
    ax_zoom.set_xlabel("x (pixel)")
    ax_zoom.set_ylabel("y (pixel)")
    
    # Pannello virtuale nel sistema locale
    box_x, box_y, box_w, box_h = 300, 350, 180, 100
    rect_panel = patches.Rectangle((box_x, box_y), box_w, box_h, linewidth=2, edgecolor='green', facecolor='#2ca02c', alpha=0.8)
    ax_zoom.add_patch(rect_panel)
    
    cx, cy = box_x + box_w/2, box_y + box_h/2
    ax_zoom.plot(cx, cy, 'ko') 
    ax_zoom.text(cx + 15, cy - 15, r"$(x_{local}, y_{local})$", color='black', fontsize=13)

    # Freccia di connessione
    con = ConnectionPatch(xyA=(0, tile_h/2), xyB=(target_x + tile_w, target_y + tile_h/2), 
                          coordsA="data", coordsB="data",
                          axesA=ax_zoom, axesB=ax_main,
                          arrowstyle="-|>", color="red", lw=2.5)
    ax_zoom.add_artist(con)

    # Box Formule
    formula_text = (
        "Equazioni di Traslazione:\n\n"
        r"$px_{global} = x_{local} + tile\_off\_x$" + "\n" +
        r"$py_{global} = y_{local} + tile\_off\_y$"
    )
    fig.text(0.65, 0.25, formula_text, fontsize=14, 
             bbox=dict(facecolor='#ffffff', edgecolor='black', boxstyle='round,pad=1', alpha=0.95))

    ax_main.set_title("Ortomosaico - Sistema di Riferimento Globale (Pixel)", fontsize=16, fontweight='bold')
    ax_main.set_xlabel("Larghezza Globale X (pixel)")
    ax_main.set_ylabel("Altezza Globale Y (pixel)")
    
    # Regolazione layout
    plt.subplots_adjust(right=0.6) 
    
    plt.savefig(OUTPUT_TILING, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvato con successo (in proporzione reale): {OUTPUT_TILING}")

# ==============================================================================
# ESECUZIONE MAIN
# ==============================================================================
if __name__ == "__main__":
    genera_schema_tiling()