"""
plot_training.py
Legge il file metrics_all.csv generato da Detectron2 durante il training.
Il file ha header sulla prima riga ma righe con numero variabile di colonne
(Detectron2 aggiunge colonne extra per le loss delle singole query).

Output:
  - training_curves.png
  - loss_detail.png
  - ap_curve.png

Uso: python plot_training.py
     python plot_training.py percorso/metrics_all.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter1d

# ==============================================================================
# ⚙️ CONFIGURAZIONE
# ==============================================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_DEFAULT = os.path.join(BASE_DIR, "output_solar_swinLb", "metrics_all.csv")

C_TOTAL = "#2C7BB6"
C_MASK  = "#D7191C"
C_CLS   = "#F07A10"
C_BOX   = "#1A9641"
C_AP50  = "#7B2D8B"
C_AP    = "#00838F"
C_LR    = "#888888"

SMOOTH_W = 15

# ==============================================================================
# 🔧 LETTURA CSV CON RIGHE A LUNGHEZZA VARIABILE
# ==============================================================================

def carica_csv(path):
    """
    Detectron2 scrive un CSV dove:
    - La riga 1 è l'header (poche colonne: iteration, lr, time, ...)
    - Le righe successive possono avere MOLTE più colonne (loss per query)
    - Alcune righe contengono i risultati di valutazione (AP, AP50, ...)

    Strategia: legge riga per riga e costruisce un dict per ogni riga,
    usando le prime N colonne come chiavi (N = lunghezza header) e
    ignorando le colonne extra senza nome.
    """
    if not os.path.exists(path):
        print(f"❌ File non trovato: {path}")
        sys.exit(1)

    with open(path, "r") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    if not lines:
        print("❌ File vuoto.")
        sys.exit(1)

    # Prima riga = header
    header = [h.strip() for h in lines[0].split(",")]
    print(f"   Header ({len(header)} colonne): {header[:8]} ...")

    righe = []
    for i, line in enumerate(lines[1:], start=2):
        valori = line.split(",")
        row = {}
        for j, key in enumerate(header):
            if j < len(valori):
                v = valori[j].strip()
                try:
                    row[key] = float(v)
                except ValueError:
                    row[key] = v
        righe.append(row)

    df = pd.DataFrame(righe)

    # Trova la colonna iterazione (a volte si chiama 'iter' o 'step')
    iter_col = None
    for c in ["iteration", "iter", "step", "global_step"]:
        if c in df.columns:
            iter_col = c
            break

    if iter_col is None:
        # Prova a usare la prima colonna numerica come iterazione
        for c in df.columns:
            if pd.to_numeric(df[c], errors="coerce").notna().sum() > len(df) * 0.8:
                iter_col = c
                print(f"⚠️  Colonna 'iteration' non trovata, uso '{c}' come asse X")
                break

    if iter_col is None:
        print("❌ Impossibile trovare la colonna delle iterazioni.")
        sys.exit(1)

    if iter_col != "iteration":
        df = df.rename(columns={iter_col: "iteration"})

    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df = df.dropna(subset=["iteration"])
    df = df.sort_values("iteration").reset_index(drop=True)

    # Converti tutte le colonne numeriche
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"✅ CSV caricato: {len(df)} righe")
    print(f"   Colonne utili: {[c for c in df.columns if df[c].notna().sum() > 0][:15]}")
    return df


def smooth(y, w=SMOOTH_W):
    arr = np.array(y, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() < 3:
        return arr
    arr[~valid] = np.interp(np.where(~valid)[0],
                             np.where(valid)[0], arr[valid])
    return uniform_filter1d(arr, size=min(w, len(arr)), mode="nearest")


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns and df[c].notna().sum() > 2:
            return c
    return None


def setup_ax(ax, title, xlabel="Iterazione", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color="#222222")
    ax.set_xlabel(xlabel, fontsize=9, color="#444444")
    ax.set_ylabel(ylabel, fontsize=9, color="#444444")
    ax.tick_params(labelsize=8, colors="#444444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.grid(True, linestyle="--", alpha=0.35, color="#AAAAAA")
    ax.set_facecolor("#FAFAFA")


def plot_line(ax, x, y, color, label, raw_alpha=0.15, lw=2.0):
    y = np.array(y, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return
    ax.plot(np.array(x)[mask], y[mask], color=color, alpha=raw_alpha, lw=0.8)
    ax.plot(np.array(x)[mask], smooth(y[mask]), color=color, lw=lw, label=label)


# ==============================================================================
# 📊 FIGURA PRINCIPALE
# ==============================================================================

def plot_main(df, output="training_curves.png"):
    itr = df["iteration"].values

    col_total = find_col(df, ["total_loss"])
    col_mask  = find_col(df, ["loss_mask", "loss_mask_ce", "mask_loss"])
    col_cls   = find_col(df, ["loss_cls",  "loss_ce",      "cls_loss"])
    col_box   = find_col(df, ["loss_bbox", "loss_box",     "box_loss"])
    col_lr    = find_col(df, ["lr", "learning_rate"])
    col_ap    = find_col(df, ["segm/AP",   "bbox/AP",  "AP"])
    col_ap50  = find_col(df, ["segm/AP50", "bbox/AP50","AP50"])

    print(f"\n   Loss trovate → total:{col_total} mask:{col_mask} "
          f"cls:{col_cls} box:{col_box} lr:{col_lr}")
    print(f"   AP trovate   → AP:{col_ap}  AP50:{col_ap50}")

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    fig.suptitle("Curve di Apprendimento — MaskDINO (Swin-L)",
                 fontsize=14, fontweight="bold", color="#111111", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32,
                           left=0.07, right=0.97, top=0.92, bottom=0.08)

    # 1. Total Loss
    ax1 = fig.add_subplot(gs[0, 0])
    setup_ax(ax1, "Total Loss", ylabel="Loss")
    if col_total:
        plot_line(ax1, itr, df[col_total].values, C_TOTAL, "Total Loss")
        ax1.legend(fontsize=8)

    # 2. Component Losses
    ax2 = fig.add_subplot(gs[0, 1])
    setup_ax(ax2, "Loss per Componente", ylabel="Loss")
    found = False
    for col, color, label in [(col_mask, C_MASK, "Mask Loss"),
                               (col_cls,  C_CLS,  "Class Loss"),
                               (col_box,  C_BOX,  "Box Loss")]:
        if col:
            plot_line(ax2, itr, df[col].values, color, label)
            found = True
    if found:
        ax2.legend(fontsize=8)

    # 3. Learning Rate
    ax3 = fig.add_subplot(gs[0, 2])
    setup_ax(ax3, "Learning Rate Schedule", ylabel="LR")
    if col_lr:
        lr_vals = df[col_lr].values
        mask    = ~np.isnan(lr_vals.astype(float))
        ax3.plot(itr[mask], lr_vals[mask], color=C_LR, lw=1.5, label="LR")
        ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax3.legend(fontsize=8)

    # 4. AP sul validation set
    ax4 = fig.add_subplot(gs[1, 0:2])
    setup_ax(ax4, "Precision sul Validation Set (AP)", ylabel="AP (%)")
    cols_ap = [c for c in [col_ap, col_ap50] if c]
    if cols_ap:
        eval_df = df.dropna(subset=cols_ap, how="all")
    else:
        eval_df = pd.DataFrame()

    if not eval_df.empty:
        itr_e = eval_df["iteration"].values
        if col_ap50:
            ax4.plot(itr_e, eval_df[col_ap50].values,
                     color=C_AP50, lw=2.2, marker="o", ms=4, label="AP@50")
        if col_ap:
            ax4.plot(itr_e, eval_df[col_ap].values,
                     color=C_AP, lw=2.2, marker="s", ms=4,
                     label="mAP (0.50:0.95)")
            best_idx  = eval_df[col_ap].idxmax()
            best_iter = eval_df.loc[best_idx, "iteration"]
            best_val  = eval_df.loc[best_idx, col_ap]
            ax4.axvline(best_iter, color=C_AP, lw=1.2, linestyle="--", alpha=0.6)
            ax4.annotate(f"Best: {best_val:.1f}%\n@ iter {int(best_iter)}",
                         xy=(best_iter, best_val),
                         xytext=(best_iter + max(itr_e)*0.04, best_val*0.80),
                         fontsize=8, color=C_AP,
                         arrowprops=dict(arrowstyle="->", color=C_AP, lw=1))
        ax4.set_ylim(bottom=0)
        ax4.legend(fontsize=9)
    else:
        ax4.text(0.5, 0.5, "Nessun dato AP trovato\n(segm/AP, segm/AP50, ...)",
                 ha="center", va="center", transform=ax4.transAxes,
                 fontsize=9, color="#888888")

    # 5. Zoom loss finale
    ax5 = fig.add_subplot(gs[1, 2])
    setup_ax(ax5, "Total Loss — Fase Finale (ultimo 30%)", ylabel="Loss")
    if col_total:
        vals   = df[col_total].values.astype(float)
        cutoff = int(len(vals) * 0.70)
        ax5.plot(itr[cutoff:], vals[cutoff:], color=C_TOTAL, alpha=0.2, lw=0.8)
        ax5.plot(itr[cutoff:], smooth(vals[cutoff:]),
                 color=C_TOTAL, lw=2.0, label="Total Loss")
        ax5.legend(fontsize=8)

    plt.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Salvato: {output}")
    plt.close()


# ==============================================================================
# 📊 LOSS DETAIL
# ==============================================================================

def plot_loss_detail(df, output="loss_detail.png"):
    itr       = df["iteration"].values
    col_total = find_col(df, ["total_loss"])
    col_mask  = find_col(df, ["loss_mask", "loss_mask_ce", "mask_loss"])
    col_cls   = find_col(df, ["loss_cls",  "loss_ce",      "cls_loss"])
    col_box   = find_col(df, ["loss_bbox", "loss_box",     "box_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor="white")
    fig.suptitle("Analisi delle Loss — MaskDINO",
                 fontsize=12, fontweight="bold", color="#111111")

    setup_ax(axes[0], "Total Loss", ylabel="Loss")
    if col_total:
        plot_line(axes[0], itr, df[col_total].values, C_TOTAL, "Total Loss")
        axes[0].legend(fontsize=9)

    setup_ax(axes[1], "Loss per Componente", ylabel="Loss")
    for col, color, label in [(col_mask, C_MASK, "Mask Loss"),
                               (col_cls,  C_CLS,  "Class Loss"),
                               (col_box,  C_BOX,  "Box Loss")]:
        if col:
            plot_line(axes[1], itr, df[col].values, color, label)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Salvato: {output}")
    plt.close()


# ==============================================================================
# 📊 AP CURVE
# ==============================================================================

def plot_ap_curve(df, output="ap_curve.png"):
    col_ap   = find_col(df, ["segm/AP",   "bbox/AP",  "AP"])
    col_ap50 = find_col(df, ["segm/AP50", "bbox/AP50","AP50"])

    cols_ap = [c for c in [col_ap, col_ap50] if c]
    if not cols_ap:
        print("⚠️  Nessun dato AP — ap_curve.png saltato.")
        return

    eval_df = df.dropna(subset=cols_ap, how="all")
    if eval_df.empty:
        print("⚠️  Righe AP tutte NaN — ap_curve.png saltato.")
        return

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    setup_ax(ax, "Curva di Apprendimento — Average Precision", ylabel="AP (%)")

    itr_e = eval_df["iteration"].values
    if col_ap50:
        ax.plot(itr_e, eval_df[col_ap50].values,
                color=C_AP50, lw=2.5, marker="o", ms=5, label="AP@IoU=0.50")
    if col_ap:
        ax.plot(itr_e, eval_df[col_ap].values,
                color=C_AP, lw=2.5, marker="s", ms=5,
                label="mAP@IoU=0.50:0.95")
        best_idx  = eval_df[col_ap].idxmax()
        best_iter = eval_df.loc[best_idx, "iteration"]
        best_val  = eval_df.loc[best_idx, col_ap]
        ax.axvline(best_iter, color="#AAAAAA", lw=1.2, linestyle="--")
        ax.annotate(f"  Best mAP: {best_val:.1f}%\n  @ iter {int(best_iter)}",
                    xy=(best_iter, best_val),
                    xytext=(best_iter, best_val * 0.75),
                    fontsize=9, color=C_AP)

    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Salvato: {output}")
    plt.close()


# ==============================================================================
# 🚀 MAIN
# ==============================================================================

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
    print(f"\n📈 Grafici di apprendimento\n   File: {csv_path}\n")

    df = carica_csv(csv_path)

    plot_main(df,        "training_curves.png")
    plot_loss_detail(df, "loss_detail.png")
    plot_ap_curve(df,    "ap_curve.png")

    print("\n🎯 Completato:")
    print("   - training_curves.png")
    print("   - loss_detail.png")
    print("   - ap_curve.png")
