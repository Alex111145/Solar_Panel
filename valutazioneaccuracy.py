#!/usr/bin/env python
"""
evaluate.py - Script di Valutazione MaskDINO Solar Panels
==========================================================
Stampa TUTTE le metriche in modo chiaro:
  - AP     = mAP@50:95  (COCO standard, più severa)
  - AP50   = mAP@50     (più permissiva, usata da YOLO)
  - AP75   = mAP@75
  - APs/m/l = per dimensione oggetto

USO:
  python evaluate.py
  python evaluate.py --weights /path/to/model_best.pth
  python evaluate.py --weights model_best.pth --threshold 0.37
"""

import os
import sys
import glob
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

try:
    from maskdino import add_maskdino_config
except ImportError:
    print("❌ Errore: 'maskdino' non trovato. Lancia da root del progetto.")
    sys.exit(1)

# ==============================================================================
# ⚙️ CONFIGURAZIONE - Modifica questi path se necessario
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset (stesso del train)
DATASET_FOLDER_NAME = "solar_datasets_coco_nuovo"
VALID_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json")
VALID_IMG  = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid")

# Modello
YAML_CONFIG = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation", "maskdino_R50_bs16_50ep_3s.yaml")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output_solar_coconuovo")
EVAL_OUTPUT = os.path.join(BASE_DIR, "eval_results")

# ==============================================================================
# 🛠️ FUNZIONI
# ==============================================================================

def get_best_weights(output_dir, override=None):
    """Trova automaticamente il miglior checkpoint."""
    if override and os.path.exists(override):
        print(f"✅ Usando weights specificati: {override}")
        return override

    best   = os.path.join(output_dir, "model_best.pth")
    final  = os.path.join(output_dir, "model_final.pth")

    if os.path.exists(best):
        size_mb = os.path.getsize(best) / 1e6
        print(f"✅ Trovato model_best.pth ({size_mb:.1f} MB)")
        return best
    elif os.path.exists(final):
        size_mb = os.path.getsize(final) / 1e6
        print(f"⚠️  model_best.pth non trovato, uso model_final.pth ({size_mb:.1f} MB)")
        return final
    else:
        checkpoints = sorted(glob.glob(os.path.join(output_dir, "model_*.pth")))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"⚠️  Uso ultimo checkpoint trovato: {os.path.basename(latest)}")
            return latest
        print(f"❌ Nessun .pth trovato in {output_dir}")
        sys.exit(1)


def setup_cfg(weights_path, threshold=0.05):
    """
    threshold=0.05 per la valutazione: basso per non escludere detection
    borderline durante il calcolo mAP (il COCOEvaluator le gestisce internamente).
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskdino_config(cfg)
    cfg.merge_from_file(YAML_CONFIG)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

    # Numero classi (1 = solar_panel)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES    = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.MaskDINO.NUM_CLASSES     = 1

    # Soglia bassa per valutazione (il COCOEvaluator calcola la curva completa)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.DATASETS.TEST = ("solar_val_eval",)

    cfg.freeze()
    return cfg


def print_metrics_report(results):
    """
    Stampa un report chiaro con spiegazione di ogni metrica.
    results = dizionario restituito da inference_on_dataset
    """
    print("\n" + "=" * 65)
    print("📊  REPORT METRICHE DI VALUTAZIONE")
    print("=" * 65)

    # ---- SEGMENTAZIONE (quella che ti interessa) ----
    if "segm" in results:
        s = results["segm"]
        print("\n🎯  SEGMENTAZIONE ISTANZA (Masks)")
        print("-" * 65)
        print(f"  AP         (mAP@50:95) = {s.get('AP',  0):.2f}%  ← Metrica principale COCO")
        print(f"  AP50       (mAP@50)    = {s.get('AP50', 0):.2f}%  ← Usata da YOLO, più permissiva")
        print(f"  AP75       (mAP@75)    = {s.get('AP75', 0):.2f}%  ← Richiede IoU più preciso")
        print(f"  APs        (small)     = {s.get('APs',  0):.2f}%")
        print(f"  APm        (medium)    = {s.get('APm',  0):.2f}%")
        print(f"  APl        (large)     = {s.get('APl',  0):.2f}%")

        print("\n  📌 Interpretazione del tuo risultato:")
        ap    = s.get('AP',   0)
        ap50  = s.get('AP50', 0)

        if ap > 0:
            print(f"\n  La metrica che il tuo trainer salva come 'best' è:")
            print(f"  → segm/AP = {ap:.2f}% (mAP@50:95)")

            print(f"\n  Confronto benchmark:")
            if ap >= 50:
                print(f"  🟢 {ap:.1f}% mAP@50:95 → ECCELLENTE (livello paper accademici)")
            elif ap >= 35:
                print(f"  🟡 {ap:.1f}% mAP@50:95 → BUONO (già pubblicabile per dataset custom)")
            elif ap >= 20:
                print(f"  🟠 {ap:.1f}% mAP@50:95 → DISCRETO (margine di miglioramento)")
            else:
                print(f"  🔴 {ap:.1f}% mAP@50:95 → DA MIGLIORARE (vedi consigli sotto)")

            print(f"\n  Se confronti con YOLO (usa AP50):")
            print(f"  → AP50 = {ap50:.2f}% - questo è il numero da confrontare con YOLO")

    # ---- BOUNDING BOX (secondaria) ----
    if "bbox" in results:
        b = results["bbox"]
        print("\n📦  BOUNDING BOX (Detection)")
        print("-" * 65)
        print(f"  AP         (mAP@50:95) = {b.get('AP',  0):.2f}%")
        print(f"  AP50       (mAP@50)    = {b.get('AP50', 0):.2f}%")

    print("\n" + "=" * 65)
    print("💡  COME MIGLIORARE SE AP < 40%:")
    print("    1. Aumenta val set (ora hai solo 23 img → troppo poche)")
    print("    2. Rotazione 0/90/180/270° nel mapper (pannelli aerei)")
    print("    3. Backbone Swin-L al posto di ResNet-50")
    print("    4. Più iterazioni: 15000-20000 con warmup 2000")
    print("=" * 65 + "\n")


# ==============================================================================
# 🚀 MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Valutazione MaskDINO Solar Panels")
    parser.add_argument("--weights",   type=str, default=None,
                        help="Path al file .pth (default: auto-detect model_best.pth)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Score threshold per inferenza (default: 0.05)")
    parser.add_argument("--val-json",  type=str, default=VALID_JSON,
                        help=f"Path al JSON di validation (default: {VALID_JSON})")
    parser.add_argument("--val-img",   type=str, default=VALID_IMG,
                        help=f"Path alle immagini val (default: {VALID_IMG})")
    args = parser.parse_args()

    print("\n🔎  SCRIPT DI VALUTAZIONE MASKDINO - SOLAR PANELS")
    print(f"    Val JSON : {args.val_json}")
    print(f"    Val IMG  : {args.val_img}")

    # Verifica esistenza file
    if not os.path.exists(args.val_json):
        print(f"❌ JSON non trovato: {args.val_json}")
        print("   Controlla VALID_JSON in cima allo script.")
        sys.exit(1)

    # Registra dataset val
    if "solar_val_eval" in MetadataCatalog:
        MetadataCatalog.remove("solar_val_eval")
    register_coco_instances("solar_val_eval", {}, args.val_json, args.val_img)

    # Trova weights
    weights = get_best_weights(OUTPUT_DIR, override=args.weights)

    # Setup config
    cfg = setup_cfg(weights, threshold=args.threshold)

    print(f"\n🖥️  Device: {cfg.MODEL.DEVICE.upper()}")
    if cfg.MODEL.DEVICE == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Crea predictor e dataloader
    print(f"\n⏳ Caricamento modello da: {os.path.basename(weights)}")
    predictor = DefaultPredictor(cfg)

    os.makedirs(EVAL_OUTPUT, exist_ok=True)

    val_loader = build_detection_test_loader(cfg, "solar_val_eval")

    # Evaluator: calcola AP, AP50, AP75, APs, APm, APl per bbox e segm
    evaluator = COCOEvaluator(
        "solar_val_eval",
        output_dir=EVAL_OUTPUT,
        # tasks=None significa: valuta tutto (bbox + segm)
    )

    print(f"\n🚀 Avvio inferenza su validation set...")
    print(f"   Output salvato in: {EVAL_OUTPUT}\n")

    # Lancia la valutazione
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    # Stampa report leggibile
    print_metrics_report(results)

    # Salva risultati in un file di testo
    report_path = os.path.join(EVAL_OUTPUT, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write("METRICHE DI VALUTAZIONE MASKDINO\n")
        f.write("=" * 50 + "\n")
        for task, metrics in results.items():
            f.write(f"\n[{task}]\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
    print(f"📄 Report salvato in: {report_path}")


if __name__ == "__main__":
    main()
