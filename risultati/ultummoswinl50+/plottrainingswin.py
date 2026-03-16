"""
plot_training.py
================
Legge il file metrics_all.csv (JSON Lines) generato da CSVLogHook durante
il training MaskDINO con Detectron2.

Ogni riga del file è un oggetto JSON indipendente:
    {"iteration": 100, "total_loss": 5.12, "lr": 0.00001, ...}

Le righe di valutazione contengono campi come "segm/AP", "segm/AP50", ecc.

Output:
    training_curves.png   — panoramica completa (loss, LR, AP, zoom)
    loss_detail.png       — total loss + loss per componente
    ap_curve.png          — curva AP con best marker

Uso:
    python plot_training.py
    python plot_training.py percorso/metrics_all.csv
    python plot_training.py percorso/metrics.json
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d

# ─────────────────────────────────────────────────────────────────────────────
# Configurazione
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_DEFAULT = os.path.join(BASE_DIR, "output_solar_swinLb", "metrics_all.csv")

COLORS = {
    "total": "#2C7BB6",
    "mask":  "#D7191C",
    "cls":   "#F07A10",
    "box":   "#1A9641",
    "ap50":  "#7B2D8B",
    "ap":    "#00838F",
    "lr":    "#888888",
}

SMOOTH_WINDOW = 15


# ─────────────────────────────────────────────────────────────────────────────
# Lettura JSON Lines
# ─────────────────────────────────────────────────────────────────────────────

def carica_metriche(path: str) -> pd.DataFrame:
    """
    Legge un file JSON Lines (una riga = un oggetto JSON).
    Supporta anche il metrics.json nativo di Detectron2.
    """
    if not os.path.exists(path):
        print(f"❌ File non trovato: {path}")
        sys.exit(1)

    records = []
    with open(path) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"   ⚠️  Riga {i} non valida, saltata")

    if not records:
        print("❌ Nessun record valido nel file.")
        sys.exit(1)

    df = pd.DataFrame(records)

    # Identifica colonna iterazione
    iter_col = next(
        (c for c in ["iteration", "iter", "step", "global_step"] if c in df.columns),
        None,
    )
    if iter_col is None:
        print("❌ Colonna iterazione non trovata.")
        sys.exit(1)

    if iter_col != "iteration":
        df = df.rename(columns={iter_col: "iteration"})

    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df = df.dropna(subset=["iteration"]).sort_values("iteration").reset_index(drop=True)

    # Converti tutto il possibile a numerico
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n_cols = sum(1 for c in df.columns if df[c].notna().any())
    print(f"✅ Caricati {len(df)} record, {n_cols} colonne con dati")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Utilità grafiche
# ─────────────────────────────────────────────────────────────────────────────

def smooth(y: np.ndarray, w: int = SMOOTH_WINDOW) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() < 3:
        return arr
    # Interpola i NaN prima di filtrare
    arr[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], arr[valid])
    return uniform_filter1d(arr, size=min(w, len(arr)), mode="nearest")


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Restituisce la prima colonna presente con almeno 3 valori validi."""
    for c in candidates:
        if c in df.columns and df[c].notna().sum() > 2:
            return c
    return None


def setup_ax(ax, title: str, xlabel: str = "Iterazione", ylabel: str = ""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color="#222")
    ax.set_xlabel(xlabel, fontsize=9, color="#444")
    ax.set_ylabel(ylabel, fontsize=9, color="#444")
    ax.tick_params(labelsize=8, colors="#444")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#CCC")
    ax.grid(True, ls="--", alpha=0.35, color="#AAA")
    ax.set_facecolor("#FAFAFA")


def plot_line(ax, x, y, color: str, label: str, raw_alpha: float = 0.15, lw: float = 2.0):
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return
    xm, ym = np.asarray(x)[mask], y[mask]
    ax.plot(xm, ym, color=color, alpha=raw_alpha, lw=0.8)
    ax.plot(xm, smooth(ym), color=color, lw=lw, label=label)


# ─────────────────────────────────────────────────────────────────────────────
# Risoluzione colonne
# ─────────────────────────────────────────────────────────────────────────────

class Cols:
    """Risolve una volta sola i nomi colonna presenti nel DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.total = find_col(df, ["total_loss"])
        self.mask  = find_col(df, ["loss_mask", "loss_mask_ce", "mask_loss"])
        self.cls   = find_col(df, ["loss_cls", "loss_ce", "cls_loss"])
        self.box   = find_col(df, ["loss_bbox", "loss_box", "box_loss"])
        self.lr    = find_col(df, ["lr", "learning_rate"])
        self.ap    = find_col(df, ["segm/AP", "bbox/AP", "AP"])
        self.ap50  = find_col(df, ["segm/AP50", "bbox/AP50", "AP50"])

    def print_summary(self):
        print(f"   Loss  → total={self.total}  mask={self.mask}  "
              f"cls={self.cls}  box={self.box}  lr={self.lr}")
        print(f"   AP    → AP={self.ap}  AP50={self.ap50}")


# ─────────────────────────────────────────────────────────────────────────────
# Figura principale (6 pannelli)
# ─────────────────────────────────────────────────────────────────────────────

def plot_main(df: pd.DataFrame, cols: Cols, output: str = "training_curves.png"):
    itr = df["iteration"].values

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    fig.suptitle("Curve di Apprendimento — MaskDINO (Swin-L)",
                 fontsize=14, fontweight="bold", color="#111", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32,
                           left=0.07, right=0.97, top=0.92, bottom=0.08)

    # ── 1. Total Loss ──
    ax1 = fig.add_subplot(gs[0, 0])
    setup_ax(ax1, "Total Loss", ylabel="Loss")
    if cols.total:
        plot_line(ax1, itr, df[cols.total].values, COLORS["total"], "Total Loss")
        ax1.legend(fontsize=8)

    # ── 2. Loss per componente ──
    ax2 = fig.add_subplot(gs[0, 1])
    setup_ax(ax2, "Loss per Componente", ylabel="Loss")
    components = [
        (cols.mask, "mask", "Mask Loss"),
        (cols.cls,  "cls",  "Class Loss"),
        (cols.box,  "box",  "Box Loss"),
    ]
    if any(c for c, _, _ in components):
        for col, key, label in components:
            if col:
                plot_line(ax2, itr, df[col].values, COLORS[key], label)
        ax2.legend(fontsize=8)

    # ── 3. Learning Rate ──
    ax3 = fig.add_subplot(gs[0, 2])
    setup_ax(ax3, "Learning Rate Schedule", ylabel="LR")
    if cols.lr:
        lr_vals = df[cols.lr].values.astype(float)
        m = ~np.isnan(lr_vals)
        ax3.plot(itr[m], lr_vals[m], color=COLORS["lr"], lw=1.5, label="LR")
        ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax3.legend(fontsize=8)

    # ── 4. AP validation ──
    ax4 = fig.add_subplot(gs[1, 0:2])
    setup_ax(ax4, "Precision sul Validation Set (AP)", ylabel="AP (%)")
    _plot_ap_on_ax(ax4, df, cols)

    # ── 5. Zoom loss finale (ultimo 30%) ──
    ax5 = fig.add_subplot(gs[1, 2])
    setup_ax(ax5, "Total Loss — Fase Finale (ultimo 30%)", ylabel="Loss")
    if cols.total:
        vals = df[cols.total].values.astype(float)
        cutoff = int(len(vals) * 0.70)
        ax5.plot(itr[cutoff:], vals[cutoff:], color=COLORS["total"], alpha=0.2, lw=0.8)
        ax5.plot(itr[cutoff:], smooth(vals[cutoff:]),
                 color=COLORS["total"], lw=2.0, label="Total Loss")
        ax5.legend(fontsize=8)

    plt.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Salvato: {output}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Loss detail
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_detail(df: pd.DataFrame, cols: Cols, output: str = "loss_detail.png"):
    itr = df["iteration"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor="white")
    fig.suptitle("Analisi delle Loss — MaskDINO",
                 fontsize=12, fontweight="bold", color="#111")

    setup_ax(axes[0], "Total Loss", ylabel="Loss")
    if cols.total:
        plot_line(axes[0], itr, df[cols.total].values, COLORS["total"], "Total Loss")
        axes[0].legend(fontsize=9)

    setup_ax(axes[1], "Loss per Componente", ylabel="Loss")
    for col, key, label in [(cols.mask, "mask", "Mask Loss"),
                             (cols.cls,  "cls",  "Class Loss"),
                             (cols.box,  "box",  "Box Loss")]:
        if col:
            plot_line(axes[1], itr, df[col].values, COLORS[key], label)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Salvato: {output}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# AP curve
# ─────────────────────────────────────────────────────────────────────────────

def _plot_ap_on_ax(ax, df: pd.DataFrame, cols: Cols):
    """Disegna AP/AP50 su un asse — usata sia in plot_main sia in plot_ap_curve."""
    ap_cols = [c for c in (cols.ap, cols.ap50) if c]
    if not ap_cols:
        ax.text(0.5, 0.5, "Nessun dato AP trovato\n(segm/AP, segm/AP50, ...)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888")
        return

    eval_df = df.dropna(subset=ap_cols, how="all")
    if eval_df.empty:
        ax.text(0.5, 0.5, "Righe AP tutte NaN",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888")
        return

    itr_e = eval_df["iteration"].values

    if cols.ap50:
        ax.plot(itr_e, eval_df[cols.ap50].values,
                color=COLORS["ap50"], lw=2.2, marker="o", ms=4, label="AP@IoU=0.50")

    if cols.ap:
        vals = eval_df[cols.ap].values
        ax.plot(itr_e, vals, color=COLORS["ap"], lw=2.2, marker="s", ms=4,
                label="mAP@IoU=0.50:0.95")

        best_idx  = eval_df[cols.ap].idxmax()
        best_iter = eval_df.loc[best_idx, "iteration"]
        best_val  = eval_df.loc[best_idx, cols.ap]
        ax.axvline(best_iter, color=COLORS["ap"], lw=1.2, ls="--", alpha=0.6)
        ax.annotate(
            f"Best: {best_val:.1f}%\n@ iter {int(best_iter)}",
            xy=(best_iter, best_val),
            xytext=(best_iter + max(itr_e) * 0.04, best_val * 0.80),
            fontsize=8, color=COLORS["ap"],
            arrowprops=dict(arrowstyle="->", color=COLORS["ap"], lw=1),
        )

    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)


def plot_ap_curve(df: pd.DataFrame, cols: Cols, output: str = "ap_curve.png"):
    ap_cols = [c for c in (cols.ap, cols.ap50) if c]
    if not ap_cols:
        print("⚠️  Nessun dato AP — ap_curve.png saltato.")
        return

    eval_df = df.dropna(subset=ap_cols, how="all")
    if eval_df.empty:
        print("⚠️  Righe AP tutte NaN — ap_curve.png saltato.")
        return

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    setup_ax(ax, "Curva di Apprendimento — Average Precision", ylabel="AP (%)")
    _plot_ap_on_ax(ax, df, cols)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Salvato: {output}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
    print(f"\n📈 Grafici di apprendimento\n   File: {csv_path}\n")

    df   = carica_metriche(csv_path)
    cols = Cols(df)
    cols.print_summary()

    plot_main(df, cols, "training_curves.png")
    plot_loss_detail(df, cols, "loss_detail.png")
    plot_ap_curve(df, cols, "ap_curve.png")

    print("\n🎯 Completato:")
    for f in ("training_curves.png", "loss_detail.png", "ap_curve.png"):
        tag = "✅" if os.path.exists(f) else "⏭️ "
        print(f"   {tag} {f}")