"""
dataset_stats.py
Legge i file COCO JSON di train e validation e genera la tabella
di distribuzione delle istanze per classe.
Output: dataset_stats.png (tabella) + stampa a terminale
"""

import json
import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ==============================================================================
# ⚙️ CONFIGURAZIONE — modifica questi path
# ==============================================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

TRAIN_JSON = os.path.join(BASE_DIR, "datasets", "solar_datasets_nuoveannotazioni",
                          "train", "_annotations.coco.json")
VALID_JSON = os.path.join(BASE_DIR, "datasets", "solar_datasets_nuoveannotazioni",
                          "valid", "_annotations.coco.json")

OUTPUT_PNG = "dataset_stats.png"

# ==============================================================================
# 🔧 FUNZIONI
# ==============================================================================

def carica_json(path, nome):
    if not os.path.exists(path):
        print(f"❌ File non trovato: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        data = json.load(f)
    print(f"✅ {nome}: {len(data['annotations'])} annotazioni, "
          f"{len(data['images'])} immagini, "
          f"{len(data['categories'])} classi")
    return data


def conta_per_classe(data):
    """Restituisce dict {category_id: count}"""
    conteggio = defaultdict(int)
    for ann in data["annotations"]:
        conteggio[ann["category_id"]] += 1
    return conteggio


def get_categorie(data):
    """Restituisce dict {id: nome}"""
    return {c["id"]: c["name"] for c in data["categories"]}


def genera_tabella(categorie, train_counts, val_counts):
    """Stampa la tabella a terminale e restituisce i dati."""
    totale_train = sum(train_counts.values())
    totale_val   = sum(val_counts.values())
    totale_tutto = totale_train + totale_val

    print("\n" + "="*70)
    print(f"{'Classe':<30} {'Train':>10} {'Val':>10} {'Incidenza %':>12}")
    print("="*70)

    righe = []
    for cat_id, nome in sorted(categorie.items()):
        t = train_counts.get(cat_id, 0)
        v = val_counts.get(cat_id, 0)
        inc = (t + v) / totale_tutto * 100 if totale_tutto > 0 else 0
        print(f"{nome:<30} {t:>10,} {v:>10,} {inc:>11.1f}%")
        righe.append((nome, t, v, inc))

    print("-"*70)
    print(f"{'Totale Istanze Annotate':<30} {totale_train:>10,} "
          f"{totale_val:>10,} {'100%':>12}")
    print("="*70)

    return righe, totale_train, totale_val


def genera_figura(righe, totale_train, totale_val, output):
    """Genera la tabella come PNG in stile LaTeX booktabs."""

    nomi    = [r[0] for r in righe]
    trains  = [r[1] for r in righe]
    vals    = [r[2] for r in righe]
    incs    = [f"{r[3]:.1f}%" for r in righe]

    # Aggiungi riga totale
    nomi.append("Totale Istanze Annotate")
    trains.append(totale_train)
    vals.append(totale_val)
    incs.append("100%")

    n_righe = len(nomi)
    fig, ax = plt.subplots(figsize=(11, 0.55 * n_righe + 1.8))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    col_labels = ["Classe di Anomalia", "Istanze (Train)", "Istanze (Val)", "Incidenza Totale"]
    col_widths  = [0.40, 0.18, 0.18, 0.18]
    col_x       = [0.02, 0.44, 0.62, 0.80]
    ROW_H       = 0.85 / n_righe
    HEADER_Y    = 0.88

    # Linea superiore (toprule)
    ax.axhline(y=HEADER_Y + 0.06, xmin=0.01, xmax=0.99,
               color="black", linewidth=1.5)

    # Header
    for j, label in enumerate(col_labels):
        ax.text(col_x[j], HEADER_Y, label,
                ha="left", va="center", fontsize=10,
                fontweight="bold", color="#111111")

    # Linea sotto header (midrule)
    ax.axhline(y=HEADER_Y - 0.06, xmin=0.01, xmax=0.99,
               color="black", linewidth=0.8)

    # Righe dati
    for i, (nome, t, v, inc) in enumerate(zip(nomi, trains, vals, incs)):
        y = HEADER_Y - 0.12 - i * ROW_H
        is_total = (i == n_righe - 1)

        # Sfondo alternato (solo righe dati, non totale)
        if not is_total and i % 2 == 1:
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.01, y - ROW_H * 0.45), 0.98, ROW_H * 0.9,
                boxstyle="square,pad=0", linewidth=0,
                facecolor="#F5F5F5", zorder=0
            ))

        bold = is_total
        fs   = 9.5

        valori = [nome, f"{t:,}", f"{v:,}", inc]
        for j, val in enumerate(valori):
            ax.text(col_x[j], y, val,
                    ha="left", va="center",
                    fontsize=fs,
                    fontweight="bold" if bold else "normal",
                    color="#111111")

        # Linea sopra il totale (bottomrule anticipato)
        if is_total:
            ax.axhline(y=y + ROW_H * 0.5, xmin=0.01, xmax=0.99,
                       color="black", linewidth=0.8)

    # Linea finale (bottomrule)
    last_y = HEADER_Y - 0.12 - (n_righe - 1) * ROW_H
    ax.axhline(y=last_y - ROW_H * 0.5, xmin=0.01, xmax=0.99,
               color="black", linewidth=1.5)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n✅ Tabella salvata: {output}")


# ==============================================================================
# 🚀 MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n📊 Analisi distribuzione dataset\n")

    train_data = carica_json(TRAIN_JSON, "TRAIN")
    val_data   = carica_json(VALID_JSON, "VALID")

    # Unisci categorie da entrambi i file
    categorie = get_categorie(train_data)
    categorie.update(get_categorie(val_data))

    train_counts = conta_per_classe(train_data)
    val_counts   = conta_per_classe(val_data)

    righe, tot_train, tot_val = genera_tabella(categorie, train_counts, val_counts)
    genera_figura(righe, tot_train, tot_val, OUTPUT_PNG)
