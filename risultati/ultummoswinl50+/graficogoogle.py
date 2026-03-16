"""
pipeline_chart.py
Genera il diagramma a blocchi della pipeline di post-processing.
Output: pipeline_overview.png (300 DPI, sfondo bianco)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ==========================================================================================
# DATI DEI BLOCCHI
# ==========================================================================================

blocks = [
    {
        "title": "Inferenza\nMaskDINO",
        "subtitle": "Maschere + BBox",
        "color": "#4A90D9",
    },
    {
        "title": "Proiezione\nAffine (TFW)",
        "subtitle": "Pixel → WGS 84",
        "color": "#5BA85A",
    },
    {
        "title": "Global NMS\nDeduplicazione",
        "subtitle": "Soglia: 1 metro",
        "color": "#E08A2F",
    },
    {
        "title": "Calcolo Area\n(GSD)",
        "subtitle": "Area in m²",
        "color": "#9B59B6",
    },
    {
        "title": "Export\nKMZ / CSV",
        "subtitle": "Navigazione campo",
        "color": "#C0392B",
    },
]

# ==========================================================================================
# LAYOUT
# ==========================================================================================

fig, ax = plt.subplots(figsize=(16, 4))
ax.set_xlim(0, 16)
ax.set_ylim(0, 4)
ax.axis("off")
fig.patch.set_facecolor("white")

BOX_W    = 2.4
BOX_H    = 1.4
BOX_Y    = 1.5       # centro verticale dei box
ARROW_Y  = BOX_Y + 0.05
GAP      = (16 - len(blocks) * BOX_W) / (len(blocks) + 1)  # spazio tra i box

# ==========================================================================================
# DISEGNO BLOCCHI E FRECCE
# ==========================================================================================

box_centers_x = []

for i, block in enumerate(blocks):
    x_left = GAP + i * (BOX_W + GAP)
    x_center = x_left + BOX_W / 2
    box_centers_x.append(x_center)

    # Ombra
    shadow = mpatches.FancyBboxPatch(
        (x_left + 0.07, BOX_Y - BOX_H / 2 - 0.07),
        BOX_W, BOX_H,
        boxstyle="round,pad=0.1",
        linewidth=0,
        facecolor="#cccccc",
        zorder=1
    )
    ax.add_patch(shadow)

    # Box principale
    box = mpatches.FancyBboxPatch(
        (x_left, BOX_Y - BOX_H / 2),
        BOX_W, BOX_H,
        boxstyle="round,pad=0.1",
        linewidth=1.5,
        edgecolor="white",
        facecolor=block["color"],
        zorder=2
    )
    ax.add_patch(box)

    # Numero fase
    ax.text(
        x_left + 0.18, BOX_Y + BOX_H / 2 - 0.18,
        str(i + 1),
        ha="center", va="center",
        fontsize=8, fontweight="bold",
        color="white", zorder=3
    )

    # Titolo
    ax.text(
        x_center, BOX_Y + 0.15,
        block["title"],
        ha="center", va="center",
        fontsize=9.5, fontweight="bold",
        color="white", zorder=3,
        linespacing=1.4
    )

    # Sottotitolo
    ax.text(
        x_center, BOX_Y - BOX_H / 2 + 0.28,
        block["subtitle"],
        ha="center", va="center",
        fontsize=7.5, fontstyle="italic",
        color="white", alpha=0.88, zorder=3
    )

    # Freccia verso il blocco successivo
    if i < len(blocks) - 1:
        x_arrow_start = x_left + BOX_W + 0.05
        x_arrow_end   = x_left + BOX_W + GAP - 0.05
        ax.annotate(
            "",
            xy=(x_arrow_end, ARROW_Y),
            xytext=(x_arrow_start, ARROW_Y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#444444",
                lw=2.0,
                mutation_scale=18
            ),
            zorder=4
        )

# ==========================================================================================
# TITOLO E FOOTER
# ==========================================================================================

ax.text(
    8, 3.65,
    "Pipeline di Post-Processing — MaskDINO → Navigazione GPS",
    ha="center", va="center",
    fontsize=12, fontweight="bold",
    color="#222222"
)

ax.text(
    8, 0.18,
    "inference.py  ·  Output: KMZ navigabile + CSV georiferito + Mosaico annotato",
    ha="center", va="center",
    fontsize=8, color="#666666", fontstyle="italic"
)

# ==========================================================================================
# SALVATAGGIO
# ==========================================================================================

OUTPUT = "pipeline_overview.png"
plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"✅ Salvato: {OUTPUT}")
plt.show()