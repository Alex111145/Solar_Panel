import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch

# ==============================================================================
# ⚙️ CONFIGURAZIONE
# ==============================================================================
TIF_PATH = "ortomosaico.tif"
OUTPUT_TILING = "schema_tiling.png"

def genera_schema_tiling():
    print("⏳ Generazione Schema Tiling in corso...")
    
    fig, ax_main = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('white')
    
    try:
        with rasterio.open(TIF_PATH) as src:
            img_w, img_h = src.width, src.height
            scale_factor = 0.05  
            h_small = int(img_h * scale_factor)
            w_small = int(img_w * scale_factor)
            img = src.read(1, out_shape=(h_small, w_small)) 
            ax_main.imshow(img, cmap='gray', alpha=0.75, extent=[0, img_w, img_h, 0])
    except Exception as e:
        img_w, img_h = 4784, 3029
        ax_main.imshow(np.random.rand(10, 10), cmap='gray', alpha=0.2, extent=[0, img_w, img_h, 0])

    tile_w, tile_h = 800, 800
    target_x, target_y = 1600, 800 # Patch selezionata
    
    # Riquadro rosso sulla mappa globale
    rect_tile = patches.Rectangle((target_x, target_y), tile_w, tile_h, 
                                  linewidth=3, edgecolor='red', facecolor='yellow', alpha=0.5)
    ax_main.add_patch(rect_tile)
    
    # Sistema locale (Zoom)
    ax_zoom = fig.add_axes([0.65, 0.45, 0.30, 0.40]) 
    ax_zoom.set_title(f"Sistema di Riferimento Locale ({tile_w}x{tile_h})", fontsize=11, fontweight='bold')
    ax_zoom.set_xlim(0, tile_w)
    ax_zoom.set_ylim(tile_h, 0) 
    ax_zoom.set_facecolor('#f8f9fa')
    ax_zoom.grid(True, linestyle='--', alpha=0.7)
    
    # 🔥 PANNELLO ABBASSATO (Inclinazione antioraria, verde)
    polygon_points = [[350, 600], [450, 580], [470, 780], [370, 800]]
    poly_panel = patches.Polygon(polygon_points, closed=True, linewidth=2, edgecolor='green', facecolor='#2ca02c', alpha=0.8)
    ax_zoom.add_patch(poly_panel)
    
    # Centroide (x=410, y=690)
    cx, cy = 410, 690
    ax_zoom.plot(cx, cy, 'ko', markersize=8, markeredgecolor='white') 
    ax_zoom.text(cx + 20, cy, r"$(x_{local}, y_{local})$", color='black', fontsize=13)

    # Legenda Centroide
    legend_text = "Centroide pannello rilevato\n"
    legend_text += f"Centroide (x, y): ({cx}, {cy})"
    ax_zoom.text(0.95, 0.95, legend_text, transform=ax_zoom.transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))

    # Freccia di connessione
    con = ConnectionPatch(xyA=(0, tile_h/2), xyB=(target_x + tile_w, target_y + tile_h/2), 
                          coordsA="data", coordsB="data", axesA=ax_zoom, axesB=ax_main,
                          arrowstyle="-|>", color="red", lw=2.5)
    ax_zoom.add_artist(con)

    plt.savefig(OUTPUT_TILING, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvato: {OUTPUT_TILING}")

if __name__ == "__main__":
    genera_schema_tiling()
