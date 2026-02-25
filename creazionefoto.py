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
# 1. GENERAZIONE SCHEMA TILING (Pannello Abbassato)
# ==============================================================================
def genera_schema_tiling():
    print("⏳ Generazione Schema Tiling (Lettura dimensioni reali in corso)...")
    
    fig, ax_main = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('white')
    
    # Lettura TIF
    try:
        with rasterio.open(TIF_PATH) as src:
            img_w, img_h = src.width, src.height
            print(f"📊 Dimensioni Reali Mosaico: {img_w} x {img_h} pixel")
            scale_factor = 0.05  
            h_small = int(img_h * scale_factor)
            w_small = int(img_w * scale_factor)
            img = src.read(1, out_shape=(h_small, w_small)) 
            ax_main.imshow(img, cmap='gray', alpha=0.75, extent=[0, img_w, img_h, 0])
    except Exception as e:
        print(f"⚠️ Impossibile caricare il TIF ({e}). Uso coordinate di simulazione.")
        img_w, img_h = 4784, 3029
        ax_main.imshow(np.random.rand(10, 10), cmap='gray', alpha=0.2, extent=[0, img_w, img_h, 0])

    # Griglia a 800 pixel
    tile_w, tile_h = 800, 800
    for x in range(0, img_w, tile_w):
        ax_main.axvline(x, color='#4169E1', linestyle='-', linewidth=0.3, alpha=0.5)
    for y in range(0, img_h, tile_h):
        ax_main.axhline(y, color='#4169E1', linestyle='-', linewidth=0.3, alpha=0.5)

    # Riquadro Patch
    target_x = 1600
    target_y = 800
    rect_tile = patches.Rectangle((target_x, target_y), tile_w, tile_h, 
                                  linewidth=3, edgecolor='red', facecolor='yellow', alpha=0.5)
    ax_main.add_patch(rect_tile)
    
    # Etichetta Patch
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9)
    text_offset_y = img_h * 0.03 
    ax_main.text(target_x, target_y - text_offset_y, f"Patch: tile_col_{target_x}_row_{target_y}", 
                 color='red', fontsize=12, fontweight='bold', bbox=bbox_props)

    # Box Zoom (Sistema locale)
    ax_zoom = fig.add_axes([0.65, 0.45, 0.30, 0.40]) 
    ax_zoom.set_title(f"Sistema di Riferimento Locale ({tile_w}x{tile_h})", fontsize=11, fontweight='bold')
    ax_zoom.set_xlim(0, tile_w)
    ax_zoom.set_ylim(tile_h, 0) 
    ax_zoom.set_facecolor('#f8f9fa')
    
    ax_zoom.set_xticks(np.arange(0, tile_w+1, 200))
    ax_zoom.set_yticks(np.arange(0, tile_h+1, 200))
    ax_zoom.grid(True, linestyle='--', alpha=0.7)
    ax_zoom.set_xlabel("x (pixel)")
    ax_zoom.set_ylabel("y (pixel)")
    
    # 🔥 MODIFICA: Poligono abbassato (Y aumentata per stare in basso)
    polygon_points = [[350, 600], [450, 580], [470, 780], [370, 800]]
    poly_panel = patches.Polygon(polygon_points, closed=True, linewidth=2, edgecolor='green', facecolor='#2ca02c', alpha=0.8)
    ax_zoom.add_patch(poly_panel)
    
    # Centroide e testo aggiornati
    cx, cy = 410, 690
    ax_zoom.plot(cx, cy, 'ko', markersize=8, markeredgecolor='white') 
    ax_zoom.text(cx + 20, cy, r"$(x_{local}, y_{local})$", color='black', fontsize=13, verticalalignment='center')

    # Legenda (Solo Centroide) in alto a destra
    legend_text = "Centroide pannello rilevato\n"
    legend_text += f"Centroide (x, y): ({cx}, {cy})"
    
    ax_zoom.text(0.95, 0.95, legend_text, transform=ax_zoom.transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))

    # Freccia di connessione
    con = ConnectionPatch(xyA=(0, tile_h/2), xyB=(target_x + tile_w, target_y + tile_h/2), 
                          coordsA="data", coordsB="data",
                          axesA=ax_zoom, axesB=ax_main,
                          arrowstyle="-|>", color="red", lw=2.5)
    ax_zoom.add_artist(con)

    ax_main.set_title("Ortomosaico - Sistema di Riferimento Globale (Pixel)", fontsize=16, fontweight='bold')
    ax_main.set_xlabel("Larghezza Globale X (pixel)")
    ax_main.set_ylabel("Altezza Globale Y (pixel)")
    
    plt.subplots_adjust(right=0.6) 
    plt.savefig(OUTPUT_TILING, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvato con successo (Inclinazione corretta e abbassata): {OUTPUT_TILING}")

# ==============================================================================
# 2. GENERAZIONE SCHEMA TRASFORMAZIONE AFFINE (Figura B)
# ==============================================================================
def genera_schema_affine():
    print("⏳ Generazione Schema Affine in corso...")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    ax.axis('off') 

    box_raster = patches.FancyBboxPatch((0.1, 0.4), 0.2, 0.3, boxstyle="round,pad=0.05", edgecolor='blue', facecolor='#e6f2ff', lw=2)
    ax.add_patch(box_raster)
    ax.text(0.2, 0.6, "Spazio Raster\n(Pixel)", ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0.2, 0.5, r"$(px, py)$", ha='center', va='center', fontsize=12)

    box_geo = patches.FancyBboxPatch((0.7, 0.4), 0.2, 0.3, boxstyle="round,pad=0.05", edgecolor='green', facecolor='#e6ffe6', lw=2)
    ax.add_patch(box_geo)
    ax.text(0.8, 0.6, "Spazio Geografico\n(WGS 84)", ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0.8, 0.5, r"$(Lon, Lat)$", ha='center', va='center', fontsize=12)

    ax.annotate("", xy=(0.7, 0.55), xytext=(0.3, 0.55), arrowprops=dict(arrowstyle="simple", color="black", lw=2))
    
    math_text = (
        "Matrice World File (.tfw)\n\n"
        r"$Lon = A \cdot px + B \cdot py + C$" + "\n" +
        r"$Lat = D \cdot px + E \cdot py + F$"
    )
    ax.text(0.5, 0.65, math_text, ha='center', va='bottom', fontsize=14, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.savefig(OUTPUT_AFFINE, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvato con successo: {OUTPUT_AFFINE}")

if __name__ == "__main__":
    genera_schema_tiling()
    genera_schema_affine()
