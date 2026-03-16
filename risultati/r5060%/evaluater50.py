#!/usr/bin/env python
"""
evaluate.py - Valutazione MaskDINO Solar Panels (v3)
=====================================================
Coerente con train.py v3:
  - Risoluzione test 800-1600px
  - Stampa tutte le metriche con interpretazione

Metriche COCO:
  AP     = mAP@50:95  (metrica principale)
  AP50   = mAP@50     (usata da YOLO, più permissiva)
  AP75   = mAP@75     (richiede IoU preciso — il nostro target)
  APs/m/l = per dimensione oggetto

USO:
  python evaluate.py
  python evaluate.py --weights /path/to/model_best.pth
  python evaluate.py --threshold 0.37
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
# CONFIGURAZIONE
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_NAME = "solar_datasets_nuoveannotazioni"
VALID_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json")
VALID_IMG  = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid")
YAML_CONFIG = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation", "maskdino_R50_bs16_50ep_3s.yaml")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output_solar_nuovor50")
EVAL_OUTPUT = os.path.join(BASE_DIR, "eval_results")

# ==============================================================================
# FUNZIONI
# ==============================================================================

def get_best_weights(output_dir, override=None):
    if override and os.path.exists(override):
        print(f"✅ Usando weights specificati: {override}")
        return override

    best  = os.path.join(output_dir, "model_best.pth")
    final = os.path.join(output_dir, "model_final.pth")

    if os.path.exists(best):
        size_mb = os.path.getsize(best) / 1e6
        print(f"✅ Trovato model_best.pth ({size_mb:.1f} MB)")
        return best
    elif os.path.exists(final):
        size_mb = os.path.getsize(final) / 1e6
        print(f"⚠️  Usando model_final.pth ({size_mb:.1f} MB)")
        return final
    else:
        checkpoints = sorted(glob.glob(os.path.join(output_dir, "model_*.pth")))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"⚠️  Uso ultimo checkpoint: {os.path.basename(latest)}")
            return latest
        print(f"❌ Nessun .pth trovato in {output_dir}")
        sys.exit(1)


def setup_cfg(weights_path, threshold=0.05):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskdino_config(cfg)
    cfg.merge_from_file(YAML_CONFIG)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.MODEL.ROI_HEADS.NUM_CLASSES    = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.MaskDINO.NUM_CLASSES     = 1

    # Coerente con training v3
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1600

    # Soglia bassa per valutazione: COCOEvaluator calcola la curva completa
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.DATASETS.TEST = ("solar_val_eval",)

    cfg.freeze()
    return cfg


def print_metrics_report(results):
    print("\n" + "=" * 65)
    print("📊  REPORT METRICHE DI VALUTAZIONE (v3)")
    print("=" * 65)

    if "segm" in results:
        s    = results["segm"]
        ap   = s.get('AP',   0)
        ap50 = s.get('AP50', 0)
        ap75 = s.get('AP75', 0)
        gap  = ap50 - ap75

        print(f"\n🎯  SEGMENTAZIONE ISTANZA (Masks)")
        print(f"  {'─' * 55}")
        print(f"  AP         (mAP@50:95) = {ap:.2f}%  ← Metrica principale COCO")
        print(f"  AP50       (mAP@50)    = {ap50:.2f}%  ← Usata da YOLO")
        print(f"  AP75       (mAP@75)    = {ap75:.2f}%  ← Target v3")
        print(f"  APs        (small)     = {s.get('APs', 0):.2f}%")
        print(f"  APm        (medium)    = {s.get('APm', 0):.2f}%")
        print(f"  APl        (large)     = {s.get('APl', 0):.2f}%")

        print(f"\n  📌 Diagnosi gap AP50 - AP75:")
        if gap < 2:
            print(f"  🔴 Gap = {gap:.1f}% — quasi zero (era il problema v2).")
            print(f"     Le mask sono ancora sistematicamente troncate ai bordi.")
            print(f"     → Verifica che il training v3 sia partito con resume=False.")
        elif gap < 8:
            print(f"  🟡 Gap = {gap:.1f}% — migliorato ma ancora basso.")
            print(f"     Le mask sono più precise ma c'è ancora margine.")
            print(f"     → Considera Smart Polygon su Roboflow per le annotazioni.")
        else:
            print(f"  ✅ Gap = {gap:.1f}% — nella norma. Le mask sono precise.")

        print(f"\n  📌 Performance complessiva:")
        if ap >= 52:
            print(f"  🟢 {ap:.1f}% → ECCELLENTE per ResNet-50 su dataset custom")
        elif ap >= 45:
            print(f"  🟢 {ap:.1f}% → MOLTO BUONO — obiettivo v3 raggiunto")
        elif ap >= 40:
            print(f"  🟡 {ap:.1f}% → BUONO — migliorato rispetto a v2 (38.57%)")
        elif ap >= 35:
            print(f"  🟠 {ap:.1f}% → DISCRETO — margine di miglioramento")
        else:
            print(f"  🔴 {ap:.1f}% → Non migliorato. Verifica resume=False nel training.")

        print(f"\n  Confronto con YOLO (usa AP50): {ap50:.2f}%")

    if "bbox" in results:
        b = results["bbox"]
        print(f"\n📦  BOUNDING BOX (Detection)")
        print(f"  {'─' * 55}")
        print(f"  AP    (mAP@50:95) = {b.get('AP',  0):.2f}%")
        print(f"  AP50  (mAP@50)    = {b.get('AP50', 0):.2f}%")
        print(f"  AP75  (mAP@75)    = {b.get('AP75', 0):.2f}%")

    print("\n" + "=" * 65)
    print("💡  SE AP NON È MIGLIORATO RISPETTO A v2:")
    print("    1. Verifica che train.py sia partito con resume=False")
    print("    2. Lancia diagnose.py per analizzare le annotazioni")
    print("    3. Considera Smart Polygon su Roboflow per ~100 immagini")
    print("    4. Prossimo step: backbone Swin-T per +4-8% AP")
    print("=" * 65 + "\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Valutazione MaskDINO Solar Panels v3")
    parser.add_argument("--weights",   type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--val-json",  type=str, default=VALID_JSON)
    parser.add_argument("--val-img",   type=str, default=VALID_IMG)
    args = parser.parse_args()

    print("\n🔎  VALUTAZIONE MASKDINO v3 — SOLAR PANELS")
    print(f"    Val JSON         : {args.val_json}")
    print(f"    Val IMG          : {args.val_img}")
    print(f"    Risoluzione test : 800-1600px")

    if not os.path.exists(args.val_json):
        print(f"❌ JSON non trovato: {args.val_json}")
        sys.exit(1)

    if "solar_val_eval" in MetadataCatalog:
        MetadataCatalog.remove("solar_val_eval")
    register_coco_instances("solar_val_eval", {}, args.val_json, args.val_img)

    weights = get_best_weights(OUTPUT_DIR, override=args.weights)
    cfg     = setup_cfg(weights, threshold=args.threshold)

    print(f"\n🖥️  Device: {cfg.MODEL.DEVICE.upper()}")
    if cfg.MODEL.DEVICE == "cuda":
        print(f"    GPU  : {torch.cuda.get_device_name(0)}")
        print(f"    VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\n⏳ Caricamento modello: {os.path.basename(weights)}")
    predictor = DefaultPredictor(cfg)

    os.makedirs(EVAL_OUTPUT, exist_ok=True)
    val_loader = build_detection_test_loader(cfg, "solar_val_eval")
    evaluator  = COCOEvaluator("solar_val_eval", output_dir=EVAL_OUTPUT)

    print(f"\n🚀 Avvio inferenza su validation set...")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    print_metrics_report(results)

    report_path = os.path.join(EVAL_OUTPUT, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write("METRICHE DI VALUTAZIONE MASKDINO v3\n")
        f.write("=" * 50 + "\n")
        for task, metrics in results.items():
            f.write(f"\n[{task}]\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
    print(f"📄 Report salvato in: {report_path}")


if __name__ == "__main__":
    main()