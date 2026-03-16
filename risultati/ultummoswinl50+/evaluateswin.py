#!/usr/bin/env python
"""
evaluateswin.py - Valutazione MaskDINO Swin-L @ iter 1999 (AP massima)
=======================================================================
Carica model_best.pth che corrisponde a iter 1999 — il picco AP di Swin-L.

USO:
  python evaluateswin.py                        # standard (model_best = iter 1999)
  python evaluateswin.py --weights path/to.pth  # checkpoint specifico
  python evaluateswin.py --output-dir /path     # output dir alternativo
"""

import os
import sys
import glob
import json
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
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
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_NAME = "solar_datasets_nuoveannotazioni"
VALID_JSON  = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json")
VALID_IMG   = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid")
EVAL_OUTPUT = os.path.join(BASE_DIR, "eval_results")

# Swin-L paths
SWINL_OUTPUT = os.path.join(BASE_DIR, "output_solar_swinLb")
SWINL_YAML   = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation",
                             "swin", "maskdino_swinl_bs16_50ep.yaml")
R50_OUTPUT   = os.path.join(BASE_DIR, "output_solar_nuovo40%20k")
R50_YAML     = os.path.join(BASE_DIR, "configs", "coco", "instance-segmentation",
                             "maskdino_R50_bs16_50ep_3s.yaml")

# ==============================================================================
# FUNZIONI
# ==============================================================================

def find_best_iter_from_metrics(output_dir):
    """Legge metrics.json e restituisce l'iterazione con AP massima."""
    metrics_path = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return None, None

    best_ap   = -1
    best_iter = -1
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d  = json.loads(line)
                ap = d.get("segm/AP", -1)
                it = d.get("iteration", -1)
                if ap > best_ap:
                    best_ap   = ap
                    best_iter = it
            except:
                continue

    return best_iter, best_ap


def get_weights(output_dir, override=None):
    """Restituisce model_best.pth — corrisponde all'iterazione con AP massima."""
    if override and os.path.exists(override):
        print(f"✅ Weights specificati: {override}")
        return override

    best  = os.path.join(output_dir, "model_best.pth")
    final = os.path.join(output_dir, "model_final.pth")

    if os.path.exists(best):
        size_mb = os.path.getsize(best) / 1e6
        # Trova a quale iter corrisponde
        best_iter, best_ap = find_best_iter_from_metrics(output_dir)
        if best_iter is not None:
            print(f"✅ model_.pth ({size_mb:.1f} MB)")
            print(f"   ↳ Corrisponde a iter {best_iter} — AP training: {best_ap:.2f}%")
        else:
            print(f"✅ model_best.pth ({size_mb:.1f} MB)")
        return best
    elif os.path.exists(final):
        size_mb = os.path.getsize(final) / 1e6
        print(f"⚠️  model_final.pth ({size_mb:.1f} MB) — non è il best!")
        return final
    else:
        checkpoints = sorted(glob.glob(os.path.join(output_dir, "model_*.pth")))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"⚠️  Ultimo checkpoint disponibile: {os.path.basename(latest)}")
            return latest
        print(f"❌ Nessun .pth trovato in {output_dir}")
        sys.exit(1)


def detect_config(output_dir):
    """Rileva automaticamente YAML e output_dir corretti."""
    # Controlla config_used.yaml per il backbone
    config_path = os.path.join(output_dir, "config_used.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            content = f.read()
        if "D2SwinTransformer" in content or "swinl" in content.lower():
            if os.path.exists(SWINL_YAML):
                print(f"  🏗️  Backbone rilevato: Swin-L")
                print(f"  📄 YAML: {os.path.basename(SWINL_YAML)}")
                return SWINL_YAML
    # Fallback R50
    print(f"  🏗️  Backbone rilevato: ResNet-50")
    print(f"  📄 YAML: {os.path.basename(R50_YAML)}")
    return R50_YAML


def setup_cfg(weights_path, yaml_path, threshold=0.05):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskdino_config(cfg)
    cfg.merge_from_file(yaml_path)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.MODEL.ROI_HEADS.NUM_CLASSES    = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.MaskDINO.NUM_CLASSES     = 1

    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1600

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.DATASETS.TEST = ("solar_val_eval",)

    cfg.freeze()
    return cfg


def print_metrics_report(results, best_iter=None, best_ap_train=None):
    print("\n" + "=" * 65)
    print(f"📊  REPORT METRICHE — Swin-L (Standard)")
    if best_iter:
        print(f"    Checkpoint: iter {best_iter} (AP training: {best_ap_train:.2f}%)")
    print("=" * 65)

    if "segm" in results:
        s    = results["segm"]
        ap   = s.get('AP',   0)
        ap50 = s.get('AP50', 0)
        ap75 = s.get('AP75', 0)
        apm  = s.get('APm',  0)
        apl  = s.get('APl',  0)
        gap  = ap50 - ap75

        print(f"\n🎯  SEGMENTAZIONE ISTANZA (Masks)")
        print(f"  {'─'*55}")
        print(f"  AP         (mAP@50:95) = {ap:.2f}%  ← Metrica principale COCO")
        print(f"  AP50       (mAP@50)    = {ap50:.2f}%")
        print(f"  AP75       (mAP@75)    = {ap75:.2f}%")
        print(f"  APm        (medium)    = {apm:.2f}%")
        print(f"  APl        (large)     = {apl:.2f}%")

        print(f"\n  📌 Gap AP50 - AP75: {gap:.1f}%")
        if gap < 2:
            print(f"  🔴 Gap quasi zero — mask imprecise")
        elif gap < 8:
            print(f"  🟡 Gap moderato — mask buone ma migliorabili")
        else:
            print(f"  ✅ Gap nella norma — mask precise")

        print(f"\n  📌 Performance Swin-L vs R50:")
        print(f"  {'─'*55}")
        r50_ap = 51.6
        delta  = ap - r50_ap
        print(f"  R50 finale     : {r50_ap:.1f}%")
        print(f"  Swin-L best    : {ap:.2f}%  (Δ {delta:+.1f}%)")

        if ap >= 55:
            print(f"  🟢 Swin-L supera R50 — backbone più potente confermato")
        elif ap >= 50:
            print(f"  🟡 Swin-L simile a R50 — overfitting parziale")
        else:
            print(f"  🔴 Swin-L sotto R50 — overfitting grave (dataset troppo piccolo)")

        print(f"\n  📌 Nota overfitting:")
        print(f"  Swin-L (200M param) su 1567 immagini → overfit dopo iter ~2000")
        print(f"  R50 (25M param) su 1567 immagini    → stabile fino a iter 25000")
        print(f"  → Per Swin-L servirebbero 5000+ immagini per convergenza completa")

    if "bbox" in results:
        b = results["bbox"]
        print(f"\n📦  BOUNDING BOX")
        print(f"  {'─'*55}")
        print(f"  AP    (mAP@50:95) = {b.get('AP',  0):.2f}%")
        print(f"  AP50  (mAP@50)    = {b.get('AP50', 0):.2f}%")
        print(f"  AP75  (mAP@75)    = {b.get('AP75', 0):.2f}%")

    print("\n" + "=" * 65 + "\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Valutazione MaskDINO Swin-L @ iter 1999")
    parser.add_argument("--weights",    type=str,   default=None,       help="Checkpoint specifico (default: model_best)")
    parser.add_argument("--threshold",  type=float, default=0.05,       help="Score threshold (default 0.05)")
    parser.add_argument("--val-json",   type=str,   default=VALID_JSON, help="Path JSON validazione")
    parser.add_argument("--val-img",    type=str,   default=VALID_IMG,  help="Path immagini validazione")
    parser.add_argument("--output-dir", type=str,   default=None,       help="Cartella output training")
    args = parser.parse_args()

    output_dir = args.output_dir or (SWINL_OUTPUT if os.path.exists(SWINL_OUTPUT) else R50_OUTPUT)

    print("\n🔎  VALUTAZIONE MASKDINO — Swin-L @ iter 1999 (best AP)")
    print(f"    Output dir : {output_dir}")
    print(f"    Val JSON   : {args.val_json}")

    if not os.path.exists(args.val_json):
        print(f"❌ JSON non trovato: {args.val_json}")
        sys.exit(1)

    # Registra dataset
    dataset_name = "solar_val_eval"
    for reg in [MetadataCatalog, DatasetCatalog]:
        if dataset_name in reg:
            reg.remove(dataset_name)
    register_coco_instances(dataset_name, {}, args.val_json, args.val_img)

    # Trova best iter da metrics.json
    best_iter, best_ap_train = find_best_iter_from_metrics(output_dir)
    if best_iter:
        print(f"\n  🏆 Best iter rilevato: {best_iter} (AP training = {best_ap_train:.2f}%)")

    weights = get_weights(output_dir, override=args.weights)
    yaml    = detect_config(output_dir)
    cfg     = setup_cfg(weights, yaml, threshold=args.threshold)

    print(f"\n🖥️  Device: {cfg.MODEL.DEVICE.upper()}")
    if cfg.MODEL.DEVICE == "cuda":
        print(f"    GPU  : {torch.cuda.get_device_name(0)}")
        print(f"    VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\n⏳ Caricamento model_best.pth (iter {best_iter})...")

    os.makedirs(EVAL_OUTPUT, exist_ok=True)

    predictor  = DefaultPredictor(cfg)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator  = COCOEvaluator(dataset_name, output_dir=EVAL_OUTPUT)

    print(f"\n🚀 Inferenza sul validation set...")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    print_metrics_report(results, best_iter=best_iter, best_ap_train=best_ap_train)

    # Salva report
    report_path = os.path.join(EVAL_OUTPUT, "metrics_swinL_best.txt")
    with open(report_path, "w") as f:
        f.write(f"METRICHE MASKDINO Swin-L @ iter {best_iter}\n")
        f.write("=" * 50 + "\n")
        for task, metrics in results.items():
            f.write(f"\n[{task}]\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
    print(f"📄 Report salvato in: {report_path}")


if __name__ == "__main__":
    main()