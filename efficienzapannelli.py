#!/usr/bin/env python
"""
efficiency_analysis.py — Analisi Termodinamica Pannelli Solari
==============================================================
Legge le patch di inferenza già prodotte da inference.py,
chiede all'utente i parametri del modulo (η_nom, γ, G, ε, T_amb),
calcola l'efficienza reale per ogni pannello rilevato e
salva le patch annotate con il valore di efficienza sovrapposto.

Se è disponibile un raster termico (.tif) co-registrato con le patch,
estrae T_cell dalla maschera via media dei pixel termici.
Altrimenti chiede T_cell manualmente o usa un valore di default.

Struttura output:
  efficiency_results/
    ├── patch_eff_<filename>.jpg   ← patch annotata
    └── report_efficienza.csv      ← tabella riepilogativa

Dipendenze:
  pip install opencv-python numpy rasterio detectron2 maskdino
"""

import os
import sys
import glob
import csv
import math
import warnings
import json

import cv2
import numpy as np

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# Detectron2 + MaskDINO (stesso setup di inference.py)
# ─────────────────────────────────────────────────────────────────
try:
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.layers import nms
    try:
        from maskdino import add_maskdino_config
        MASKDINO_OK = True
    except ImportError:
        MASKDINO_OK = False
        print("⚠️  MaskDINO non trovato. Modalità 'patch pre-annotate' disabilitata.")
except ImportError:
    torch = None
    MASKDINO_OK = False
    print("⚠️  Detectron2 non trovato. Necessario per re-inferenza sulle patch.")

# ─────────────────────────────────────────────────────────────────
# COSTANTI DI DEFAULT (sovrascrivibili via input utente)
# ─────────────────────────────────────────────────────────────────
DEFAULT_ETA_NOM  = 0.20          # 20 %
DEFAULT_GAMMA    = -0.0035       # -0.35 %/°C (policristallino)
DEFAULT_G        = 1000.0        # W/m²
DEFAULT_T_STC    = 25.0          # °C
DEFAULT_EPS      = 0.92          # emissività vetro temperato
DEFAULT_T_AMB    = 25.0          # °C ambiente
DEFAULT_T_CELL   = 45.0          # °C cella (fallback senza raster termico)

SOGLIA_DETECTION = 0.45
NMS_THRESH       = 0.40
BOOST            = 2.5

# colori (BGR)
COLOR_MASK       = (0, 200, 0)   # verde scuro — maschera OK
COLOR_HOTSPOT    = (0, 0, 255)   # rosso — degrado > 5 %
COLOR_TEXT_BG    = (0, 0, 0)
COLOR_TEXT_FG    = (255, 255, 255)

# ═══════════════════════════════════════════════════════════════════
# 1. INPUT UTENTE
# ═══════════════════════════════════════════════════════════════════

def ask_float(prompt, default):
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print("  ⚠️  Valore non valido, riprova.")


def ask_str(prompt, default=""):
    raw = input(f"  {prompt} [{default}]: ").strip()
    return raw if raw else default


def collect_user_params():
    print("\n" + "═"*60)
    print("  PARAMETRI MODULO FOTOVOLTAICO")
    print("═"*60)
    print("  Premi INVIO per usare il valore di default.\n")

    params = {}
    params["eta_nom"]  = ask_float("Efficienza nominale STC η_nom  (es. 0.20 = 20 %)", DEFAULT_ETA_NOM)
    params["gamma"]    = ask_float("Coeff. temperatura γ  (es. -0.0035 per -0.35 %/°C)", DEFAULT_GAMMA)
    params["G"]        = ask_float("Irraggiamento G  [W/m²]", DEFAULT_G)
    params["T_STC"]    = ask_float("Temperatura STC  [°C]", DEFAULT_T_STC)
    params["epsilon"]  = ask_float("Emissività vetro ε  (0.85 – 0.95)", DEFAULT_EPS)
    params["T_amb"]    = ask_float("Temperatura ambiente T_amb  [°C]", DEFAULT_T_AMB)

    print("\n" + "─"*60)
    print("  SORGENTE TEMPERATURA CELLA")
    print("  1) Raster termico (.tif) co-registrato con le patch")
    print("  2) Valore fisso (es. misurato con termometro)")
    print("  3) Valore per ogni pannello inserito manualmente")
    print("─"*60)

    src = ask_str("Sorgente temperatura [1/2/3]", "2")
    params["temp_source"] = src.strip()

    if src == "1":
        params["thermal_tif"] = ask_str(
            "Percorso raster termico  (es. thermal_map.tif)",
            "thermal_map.tif"
        )
    elif src == "2":
        params["T_cell_fixed"] = ask_float(
            "Temperatura cella fissa T_cell  [°C]", DEFAULT_T_CELL
        )
    # src == "3" → sarà chiesta per ogni pannello

    print("\n" + "─"*60)
    print("  DIRECTORY PATCH")
    print("─"*60)
    params["patches_dir"] = ask_str(
        "Directory patch inferenza  (es. inference_results)",
        "inference_results"
    )
    params["output_dir"] = ask_str(
        "Directory output annotate  (es. efficiency_results)",
        "efficiency_results"
    )

    # Opzionale: modello per re-inferenza se le patch non sono già annotate
    params["use_model"] = ask_str(
        "Eseguire re-inferenza MaskDINO sulle patch? [s/N]", "N"
    ).lower() in ("s", "si", "y", "yes")

    if params["use_model"] and MASKDINO_OK:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        _OUT_SWINL = os.path.join(BASE_DIR, "output_solar_swinL")
        _OUT_R50   = os.path.join(BASE_DIR, "output_solar_nuovo40%20k")
        default_out = _OUT_SWINL if os.path.exists(
            os.path.join(_OUT_SWINL, "last_checkpoint")
        ) else _OUT_R50
        params["model_output_dir"] = ask_str(
            "Directory pesi modello", default_out
        )
        _YAML_SWINL = os.path.join(
            BASE_DIR, "configs", "coco", "instance-segmentation",
            "swin", "maskdino_swinl_bs16_50ep.yaml"
        )
        _YAML_R50 = os.path.join(
            BASE_DIR, "configs", "coco", "instance-segmentation",
            "maskdino_R50_bs16_50ep_3s.yaml"
        )
        default_yaml = _YAML_SWINL if os.path.exists(_YAML_SWINL) else _YAML_R50
        params["yaml_config"] = ask_str("YAML config modello", default_yaml)

    return params

# ═══════════════════════════════════════════════════════════════════
# 2. MODELLO TERMODINAMICO
# ═══════════════════════════════════════════════════════════════════

def correct_emissivity(T_apparent_C, epsilon, T_amb_C):
    """
    Correzione emissività secondo Stefan-Boltzmann.
    Restituisce T_reale in °C.
    """
    T_ap_K  = T_apparent_C + 273.15
    T_amb_K = T_amb_C + 273.15
    val = (T_ap_K**4 - (1 - epsilon) * T_amb_K**4) / epsilon
    if val < 0:
        return T_apparent_C   # fallback: se fisicamente impossibile
    return val**0.25 - 273.15


def eta_reale(eta_nom, gamma, T_cell, T_STC=25.0):
    """Efficienza reale secondo modello lineare di degradazione termica."""
    return eta_nom * (1 + gamma * (T_cell - T_STC))


def potenza_reale(G, A_mask, eta_r):
    return G * A_mask * eta_r


def potenza_teorica(G, A_mask, eta_nom):
    return G * A_mask * eta_nom


def perdita_percentuale(P_teo, P_real):
    if P_teo == 0:
        return 0.0
    return (P_teo - P_real) / P_teo * 100.0


def pr_locale(eta_r, eta_nom):
    return eta_r / eta_nom if eta_nom != 0 else 0.0

# ═══════════════════════════════════════════════════════════════════
# 3. ESTRAZIONE T_CELL DA RASTER TERMICO
# ═══════════════════════════════════════════════════════════════════

def load_thermal_raster(tif_path):
    """Carica raster termico come array float32."""
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            band = src.read(1).astype(np.float32)
        print(f"  ✅ Raster termico caricato: {tif_path} ({band.shape})")
        return band
    except Exception as e:
        print(f"  ⚠️  Impossibile caricare raster termico: {e}")
        return None


def extract_temp_from_mask(thermal_arr, mask_bool):
    """Media pixel termici dentro la maschera binaria."""
    if thermal_arr is None or not mask_bool.any():
        return None
    vals = thermal_arr[mask_bool]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) > 0 else None

# ═══════════════════════════════════════════════════════════════════
# 4. RE-INFERENZA MASKDINO (opzionale)
# ═══════════════════════════════════════════════════════════════════

def get_best_weights(output_dir):
    for name in ("model_best.pth", "model_final.pth"):
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            return p
    ckpts = glob.glob(os.path.join(output_dir, "model_*.pth"))
    return max(ckpts, key=os.path.getctime) if ckpts else None


def build_predictor(params):
    if not MASKDINO_OK or torch is None:
        return None
    weights = get_best_weights(params["model_output_dir"])
    if not weights:
        print("  ❌ Nessun checkpoint trovato.")
        return None
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskdino_config(cfg)
    cfg.merge_from_file(params["yaml_config"])
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES    = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.MaskDINO.NUM_CLASSES     = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST         = SOGLIA_DETECTION
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.3
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1600
    print(f"  ✅ Modello caricato: {os.path.basename(weights)}")
    return DefaultPredictor(cfg)


def run_inference(predictor, img):
    """Restituisce liste (masks, boxes, scores)."""
    if predictor is None:
        return [], [], []
    outputs   = predictor(img)
    instances = outputs["instances"].to("cpu")
    if len(instances) == 0:
        return [], [], []
    # boost + soglia
    instances._fields["scores"] = torch.clamp(instances.scores * BOOST, max=1.0)
    mask = instances.scores >= SOGLIA_DETECTION
    instances = instances[mask]
    if len(instances) == 0:
        return [], [], []
    keep      = nms(instances.pred_boxes.tensor, instances.scores, NMS_THRESH)
    instances = instances[keep]
    masks  = instances.pred_masks.numpy()
    boxes  = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    return masks, boxes, scores

# ═══════════════════════════════════════════════════════════════════
# 5. ANNOTAZIONE PATCH
# ═══════════════════════════════════════════════════════════════════

def annotate_patch(img_bgr, masks, boxes, scores, eff_values, loss_values):
    """
    Disegna maschera + testo efficienza su ogni pannello rilevato.
    eff_values  : lista float (η_reale %)
    loss_values : lista float (perdita %)
    """
    annotated = img_bgr.copy()

    for k, (mask, box, score, eta_pct, loss_pct) in enumerate(
        zip(masks, boxes, scores, eff_values, loss_values)
    ):
        # Colore maschera: rosso se perdita > 5 %, verde altrimenti
        color = COLOR_HOTSPOT if loss_pct > 5.0 else COLOR_MASK

        # Overlay semitrasparente della maschera
        overlay        = annotated.copy()
        overlay[mask]  = color
        annotated      = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

        # Contorno
        mask_u8 = mask.astype("uint8") * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, color, 2)

        # Centroide per il testo
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
        else:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)

        # Testo: efficienza reale + score
        label_eta   = f"η={eta_pct:.1f}%"
        label_score = f"conf={score:.0%}"
        label_loss  = f"Δ={loss_pct:+.1f}%"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(img_bgr.shape[0] / 1200, 0.75))
        thickness  = 1

        for idx, label in enumerate([label_eta, label_loss, label_score]):
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            tx = max(0, cx - tw // 2)
            ty = cy - 10 + idx * int(th * 1.6)
            ty = max(th + 4, min(ty, annotated.shape[0] - 4))

            # sfondo testo
            cv2.rectangle(
                annotated,
                (tx - 2, ty - th - 2),
                (tx + tw + 2, ty + 2),
                COLOR_TEXT_BG, -1
            )
            # testo
            cv2.putText(
                annotated, label, (tx, ty),
                font, font_scale, COLOR_TEXT_FG, thickness, cv2.LINE_AA
            )

    return annotated


def draw_legend(img_bgr, params):
    """Aggiunge un box informativo con i parametri in alto a sinistra."""
    lines = [
        f"η_nom={params['eta_nom']*100:.1f}%  γ={params['gamma']*100:.3f}%/°C",
        f"G={params['G']:.0f} W/m²  T_STC={params['T_STC']:.0f}°C",
        f"ε={params['epsilon']:.2f}  T_amb={params['T_amb']:.1f}°C",
    ]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness  = 1
    pad        = 6
    (_, th), _ = cv2.getTextSize("A", font, font_scale, thickness)
    box_h      = (th + pad) * len(lines) + pad
    box_w      = 300
    cv2.rectangle(img_bgr, (0, 0), (box_w, box_h), (30, 30, 30), -1)
    for i, line in enumerate(lines):
        y = pad + (i + 1) * (th + pad)
        cv2.putText(img_bgr, line, (pad, y), font, font_scale,
                    (200, 200, 200), thickness, cv2.LINE_AA)
    return img_bgr

# ═══════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*60)
    print("  ANALISI EFFICIENZA TERMODINAMICA — Pannelli Solari")
    print("  Basato su MaskDINO Solar Pipeline")
    print("═"*60)

    # ── 6.1 Parametri utente ─────────────────────────────────────
    params = collect_user_params()

    os.makedirs(params["output_dir"], exist_ok=True)

    # ── 6.2 Raster termico (opzionale) ───────────────────────────
    thermal_arr = None
    if params["temp_source"] == "1":
        thermal_arr = load_thermal_raster(params.get("thermal_tif", ""))

    # ── 6.3 Caricamento patch ─────────────────────────────────────
    patches = sorted(
        glob.glob(os.path.join(params["patches_dir"], "*.jpg")) +
        glob.glob(os.path.join(params["patches_dir"], "*.JPG")) +
        glob.glob(os.path.join(params["patches_dir"], "*.png")) +
        glob.glob(os.path.join(params["patches_dir"], "*.PNG"))
    )
    if not patches:
        print(f"\n❌ Nessuna patch trovata in: {params['patches_dir']}")
        sys.exit(1)
    print(f"\n📂 Trovate {len(patches)} patch in '{params['patches_dir']}'")

    # ── 6.4 Predictor (opzionale) ─────────────────────────────────
    predictor = None
    if params.get("use_model") and MASKDINO_OK:
        print("\n🔧 Caricamento modello MaskDINO...")
        predictor = build_predictor(params)

    # ── 6.5 CSV report ────────────────────────────────────────────
    csv_path = os.path.join(params["output_dir"], "report_efficienza.csv")
    csv_fields = [
        "patch", "pannello_id", "score",
        "T_apparent_C", "T_reale_C",
        "eta_reale_pct", "P_reale_W", "P_teo_W",
        "perdita_pct", "PR_locale"
    ]
    csv_rows = []

    # ── 6.6 Loop patch ────────────────────────────────────────────
    total_panels   = 0
    total_loss_W   = 0.0
    A_mask_default = 1.70   # m² — stimato se non disponibile

    for i, patch_path in enumerate(patches):
        fname = os.path.basename(patch_path)
        img   = cv2.imread(patch_path)
        if img is None:
            print(f"  ⚠️  Impossibile leggere: {fname}")
            continue

        # Inferenza o detection nulla (patch già annotate senza maschera)
        if predictor is not None:
            masks, boxes, scores = run_inference(predictor, img)
        else:
            # Nessun modello: chiedi quanti pannelli ci sono nella patch
            print(f"\n─── Patch {i+1}/{len(patches)}: {fname} ───")
            if params["temp_source"] != "3":
                n_str = input(
                    f"  Quanti pannelli rilevati in questa patch? "
                    f"(0 = salta): "
                ).strip()
                n = int(n_str) if n_str.isdigit() else 0
            else:
                n = 1   # almeno 1 per chiedere T
            # Senza maschera, usiamo box fittizi centrati
            h, w = img.shape[:2]
            masks  = [np.ones((h, w), dtype=bool)] * n
            boxes  = [np.array([0, 0, w, h])] * n
            scores = [1.0] * n

        if len(masks) == 0:
            print(f"  [{i+1:3d}/{len(patches)}] {fname:<40} | Pannelli: 0 — saltato")
            continue

        eff_values  = []
        loss_values = []

        for k in range(len(masks)):
            # ── Temperatura cella ─────────────────────────────────
            T_apparent = None

            if params["temp_source"] == "1" and thermal_arr is not None:
                mask_bool = masks[k]
                # Ridimensiona maschera se necessario
                if mask_bool.shape != thermal_arr.shape:
                    mask_bool = cv2.resize(
                        mask_bool.astype("uint8"),
                        (thermal_arr.shape[1], thermal_arr.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                T_apparent = extract_temp_from_mask(thermal_arr, mask_bool)

            if T_apparent is None and params["temp_source"] == "2":
                T_apparent = params["T_cell_fixed"]

            if T_apparent is None and params["temp_source"] == "3":
                T_apparent = ask_float(
                    f"  Patch {fname} | Pannello {k+1} — T_cell apparente [°C]",
                    DEFAULT_T_CELL
                )

            if T_apparent is None:
                T_apparent = DEFAULT_T_CELL

            # Correzione emissività
            T_reale = correct_emissivity(
                T_apparent, params["epsilon"], params["T_amb"]
            )

            # Efficienza reale
            eta_r   = eta_reale(params["eta_nom"], params["gamma"], T_reale, params["T_STC"])
            eta_pct = eta_r * 100.0

            # Potenze (area stimata o da maschera pixel)
            A = A_mask_default
            P_real = potenza_reale(params["G"], A, eta_r)
            P_teo  = potenza_teorica(params["G"], A, params["eta_nom"])
            loss   = perdita_percentuale(P_teo, P_real)
            PR     = pr_locale(eta_r, params["eta_nom"])

            eff_values.append(eta_pct)
            loss_values.append(loss)
            total_loss_W += (P_teo - P_real)

            score_val = float(scores[k]) if hasattr(scores[k], "__float__") else scores[k]
            csv_rows.append({
                "patch":         fname,
                "pannello_id":   k + 1,
                "score":         round(score_val, 4),
                "T_apparent_C":  round(T_apparent, 2),
                "T_reale_C":     round(T_reale, 2),
                "eta_reale_pct": round(eta_pct, 2),
                "P_reale_W":     round(P_real, 2),
                "P_teo_W":       round(P_teo, 2),
                "perdita_pct":   round(loss, 2),
                "PR_locale":     round(PR, 4),
            })

        total_panels += len(masks)

        # Annotazione patch
        annotated = annotate_patch(img, masks, boxes, scores, eff_values, loss_values)
        annotated = draw_legend(annotated, params)

        out_name = f"patch_eff_{fname}"
        cv2.imwrite(os.path.join(params["output_dir"], out_name), annotated)
        avg_eff = sum(eff_values) / len(eff_values)
        print(
            f"  [{i+1:3d}/{len(patches)}] {fname:<40} | "
            f"Pannelli: {len(masks):2d} | η_avg: {avg_eff:.1f}%"
        )

    # ── 6.7 Salva CSV ────────────────────────────────────────────
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    # ── 6.8 Riepilogo ────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  RIEPILOGO ANALISI EFFICIENZA")
    print("═"*60)
    print(f"  Patch analizzate   : {len(patches)}")
    print(f"  Pannelli totali    : {total_panels}")
    if csv_rows:
        etas = [r["eta_reale_pct"] for r in csv_rows]
        losses = [r["perdita_pct"]  for r in csv_rows]
        print(f"  η_reale media      : {sum(etas)/len(etas):.2f} %")
        print(f"  Perdita media/mod  : {sum(losses)/len(losses):.2f} %")
        print(f"  Perdita totale est : {total_loss_W:.1f} W")
    print(f"\n  Patch annotate → {params['output_dir']}/")
    print(f"  Report CSV         → {csv_path}")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
