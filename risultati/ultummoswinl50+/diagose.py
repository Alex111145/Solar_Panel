#!/usr/bin/env python
"""
fix_annotations.py - Filtra annotazioni border e diagnostica qualità dataset
=============================================================================
PROBLEMA:
  La diagnostica ha rilevato che il 52% delle annotazioni (5726/10923)
  rappresenta pannelli TRONCATI al bordo del tile (aspect ratio ~0.1,
  fill ratio ~0.50). Questi insegnano al modello che un pannello è uno
  sliver 1:10 invece di un rettangolo — causa diretta di AP bassa.

FUNZIONI:
  1. diagnose  → analisi completa senza modificare nulla (default)
  2. fix       → filtra le annotazioni border e salva backup
  3. restore   → ripristina i JSON originali dal backup

USO:
  python fix_annotations.py                  # diagnosi (dry-run)
  python fix_annotations.py fix              # applica filtraggio
  python fix_annotations.py fix --verbose    # applica + mostra ogni annotazione rimossa
  python fix_annotations.py restore          # ripristina backup
  python fix_annotations.py diagnose         # solo diagnosi (esplicita)

SOGLIE DI FILTRAGGIO (modificabili):
  AR_THRESH   = 0.25  → rimuove se w/h < 0.25 o h/w < 0.25
  FILL_THRESH = 0.55  → rimuove se polygon_area/bbox_area < 0.55
  MIN_AREA    = 200   → rimuove se area < 200 px²
"""

import os
import sys
import json
import shutil
import argparse
import math
from collections import Counter, defaultdict

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "solar_datasets_nuoveannotazioni")
TRAIN_JSON  = os.path.join(DATASET_DIR, "train", "_annotations.coco.json")
VALID_JSON  = os.path.join(DATASET_DIR, "valid", "_annotations.coco.json")

AR_THRESH   = 0.25
FILL_THRESH = 0.55
MIN_AREA    = 200

# ==============================================================================
# FUNZIONI CORE
# ==============================================================================

def load_json(path):
    if not os.path.exists(path):
        print(f"❌ Non trovato: {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)


def polygon_area(segmentation):
    if not segmentation or not isinstance(segmentation[0], list):
        return 0
    coords = segmentation[0]
    xs, ys = coords[0::2], coords[1::2]
    n = len(xs)
    if n < 3:
        return 0
    area = sum(xs[i] * ys[(i+1)%n] - xs[(i+1)%n] * ys[i] for i in range(n))
    return abs(area) / 2.0


def polygon_point_count(segmentation):
    if not segmentation or not isinstance(segmentation[0], list):
        return 0
    return max(len(seg) // 2 for seg in segmentation)


def classify_annotation(ann, ar_thresh, fill_thresh, min_area):
    """
    Restituisce (is_border, motivo) per ogni annotazione.
    """
    bbox = ann.get('bbox', [0, 0, 1, 1])
    area = ann.get('area', 0)
    seg  = ann.get('segmentation', [])
    w, h = bbox[2], bbox[3]

    if h <= 0 or w <= 0:
        return True, "bbox degenerata"
    if area < min_area:
        return True, f"area piccola ({area:.0f}px² < {min_area})"

    ar_wh, ar_hw = w/h, h/w
    if ar_wh < ar_thresh:
        return True, f"troncato verticalmente (w/h={ar_wh:.2f})"
    if ar_hw < ar_thresh:
        return True, f"troncato orizzontalmente (h/w={ar_hw:.2f})"

    poly_a = polygon_area(seg)
    bbox_a = w * h
    if poly_a > 0 and bbox_a > 0:
        fill = min(poly_a, bbox_a) / max(poly_a, bbox_a)
        if fill < fill_thresh:
            return True, f"pannello parziale (fill={fill:.2f})"

    return False, ""


# ==============================================================================
# COMANDO: DIAGNOSE
# ==============================================================================

def cmd_diagnose(args):
    print("\n" + "═"*65)
    print("  🔬 DIAGNOSTICA ANNOTAZIONI — SOLAR PANELS")
    print("═"*65)
    print(f"  Soglie: AR={AR_THRESH}  Fill={FILL_THRESH}  MinArea={MIN_AREA}px²")

    for json_path, split in [(TRAIN_JSON, "TRAINING"), (VALID_JSON, "VALIDATION")]:
        data = load_json(json_path)
        if not data:
            continue

        anns   = data.get('annotations', [])
        images = data.get('images', [])
        total  = len(anns)
        img_map = {img['id']: img.get('file_name','?') for img in images}

        print(f"\n{'─'*65}")
        print(f"  📂 {split} — {total} annotazioni, {len(images)} immagini")
        print(f"{'─'*65}")

        # Classificazione
        border, clean, reasons = [], [], []
        for ann in anns:
            is_b, reason = classify_annotation(ann, AR_THRESH, FILL_THRESH, MIN_AREA)
            if is_b:
                border.append((ann, reason))
                reasons.append(reason.split(' (')[0])  # categoria senza dettaglio
            else:
                clean.append(ann)

        n_border = len(border)
        pct      = n_border / total * 100

        print(f"\n  ⚠️  Annotazioni border (da rimuovere): {n_border} ({pct:.1f}%)")
        print(f"  ✅  Annotazioni pulite (da mantenere):  {len(clean)} ({100-pct:.1f}%)")

        # Breakdown motivi
        reason_count = Counter(reasons)
        print(f"\n  Motivi rimozione:")
        for r, c in sorted(reason_count.items(), key=lambda x: -x[1]):
            print(f"    {r:<40}: {c:5d} ({c/total*100:.1f}%)")

        # Punti per poligono
        pts_all    = [polygon_point_count(a.get('segmentation',[])) for a in anns]
        pts_clean  = [polygon_point_count(a.get('segmentation',[])) for a in clean]
        avg_all    = sum(pts_all)  / len(pts_all)  if pts_all  else 0
        avg_clean  = sum(pts_clean)/ len(pts_clean) if pts_clean else 0

        buckets = {
            "4 punti (rettangolo grezzo)":  [p for p in pts_clean if p <= 4],
            "5-8 punti (semplice)":         [p for p in pts_clean if 5 <= p <= 8],
            "9-16 punti (buono)":           [p for p in pts_clean if 9 <= p <= 16],
            "17+ punti (dettagliato)":      [p for p in pts_clean if p > 16],
        }

        print(f"\n  Qualità poligoni (annotazioni PULITE, media: {avg_clean:.1f} punti):")
        for label, vals in buckets.items():
            pct_b = len(vals)/len(pts_clean)*100 if pts_clean else 0
            bar   = "█" * int(pct_b/3)
            print(f"    {label:<35}: {len(vals):5d} ({pct_b:5.1f}%) {bar}")

        # Distribuzione aree (solo annotazioni pulite)
        areas_clean = [a.get('area',0) for a in clean]
        small  = [a for a in areas_clean if a < 1024]
        medium = [a for a in areas_clean if 1024 <= a < 9216]
        large  = [a for a in areas_clean if a >= 9216]

        print(f"\n  Distribuzione aree (post-filtraggio):")
        for label, vals, tag in [
            ("Small  (<1024px²)",  small,  "APs"),
            ("Medium (1024-9216)", medium, "APm"),
            ("Large  (>9216px²)", large,  "APl"),
        ]:
            pct_a = len(vals)/len(areas_clean)*100 if areas_clean else 0
            avg_a = sum(vals)/len(vals) if vals else 0
            print(f"    {label:<25}: {len(vals):5d} ({pct_a:5.1f}%) avg={avg_a:.0f}px²  [{tag}]")
            if tag == "APs" and len(vals) == 0:
                print(f"      ℹ️  APs=0 sarà normale (nessun oggetto small)")

        # Densità per immagine
        per_img = defaultdict(int)
        for a in clean:
            per_img[a['image_id']] += 1
        counts = list(per_img.values())
        if counts:
            print(f"\n  Densità (post-filtraggio):")
            print(f"    Immagini con annotazioni: {len(counts)}")
            print(f"    Pannelli medi/immagine  : {sum(counts)/len(counts):.1f}")
            print(f"    Max pannelli/immagine   : {max(counts)}")

        # Stima impatto
        print(f"\n  💡 Impatto atteso rimozione border:")
        print(f"    Le {n_border} annotazioni rimossa insegnavano al modello")
        print(f"    che un pannello è uno sliver 1:10 invece di un rettangolo.")
        print(f"    Stima miglioramento: +5-10% AP, separazione AP75 da AP50.")

    print(f"\n{'═'*65}")
    print(f"  Per applicare: python fix_annotations.py fix")
    print(f"  Per ripristinare: python fix_annotations.py restore")
    print(f"{'═'*65}\n")


# ==============================================================================
# COMANDO: FIX
# ==============================================================================

def cmd_fix(args):
    print("\n" + "═"*65)
    print("  🔧 APPLICAZIONE FIX — FILTRAGGIO ANNOTAZIONI BORDER")
    print("═"*65)
    print(f"  Soglie: AR={AR_THRESH}  Fill={FILL_THRESH}  MinArea={MIN_AREA}px²\n")

    total_removed = 0
    total_kept    = 0

    for json_path, split in [(TRAIN_JSON, "TRAIN"), (VALID_JSON, "VALID")]:
        data = load_json(json_path)
        if not data:
            continue

        anns  = data.get('annotations', [])
        total = len(anns)

        keep   = []
        remove = []
        for ann in anns:
            is_b, reason = classify_annotation(ann, AR_THRESH, FILL_THRESH, MIN_AREA)
            if is_b:
                remove.append((ann, reason))
            else:
                keep.append(ann)

        # Backup
        backup = json_path.replace(".json", "_backup_original.json")
        if not os.path.exists(backup):
            shutil.copy2(json_path, backup)
            print(f"  [{split}] 💾 Backup: {os.path.basename(backup)}")
        else:
            print(f"  [{split}] 💾 Backup già esistente (non sovrascritto)")

        # Verbose
        if args.verbose and remove:
            img_map = {img['id']: img.get('file_name','?') for img in data.get('images',[])}
            print(f"\n  [{split}] Prime 20 annotazioni rimosse:")
            for ann, reason in remove[:20]:
                img_name = img_map.get(ann.get('image_id'), '?')
                print(f"    ann_id {ann['id']:6d} | {img_name[:35]:<35} | {reason}")
            if len(remove) > 20:
                print(f"    ... e altri {len(remove)-20}")

        # Aggiorna JSON
        data['annotations'] = keep
        ids_with_ann   = set(a['image_id'] for a in keep)
        orig_imgs      = data.get('images', [])
        data['images'] = [img for img in orig_imgs if img['id'] in ids_with_ann]
        n_imgs_removed = len(orig_imgs) - len(data['images'])

        with open(json_path, 'w') as f:
            json.dump(data, f)

        n_rem = len(remove)
        print(f"  [{split}] Annotazioni: {total:5d} → {len(keep):5d}  "
              f"(rimosse {n_rem:4d} = {n_rem/total*100:.1f}%)")
        print(f"  [{split}] Immagini  : {len(orig_imgs):5d} → {len(data['images']):5d}  "
              f"(-{n_imgs_removed} senza annotazioni)\n")

        total_removed += n_rem
        total_kept    += len(keep)

    print(f"{'═'*65}")
    print(f"  TOTALE rimosso : {total_removed}")
    print(f"  TOTALE rimasto : {total_kept}")
    print(f"  Riduzione      : {total_removed/(total_removed+total_kept)*100:.1f}%")
    print(f"\n  ✅ Fix applicata. Ora rilancia il training da zero:")
    print(f"     python train.py")
    print(f"\n  Per annullare: python fix_annotations.py restore")
    print(f"{'═'*65}\n")


# ==============================================================================
# COMANDO: RESTORE
# ==============================================================================

def cmd_restore(args):
    print("\n🔄 Ripristino backup originali...")
    for json_path in [TRAIN_JSON, VALID_JSON]:
        backup = json_path.replace(".json", "_backup_original.json")
        if os.path.exists(backup):
            shutil.copy2(backup, json_path)
            split = "TRAIN" if "train" in json_path else "VALID"
            print(f"  ✅ [{split}] Ripristinato: {os.path.basename(json_path)}")
        else:
            split = "TRAIN" if "train" in json_path else "VALID"
            print(f"  ❌ [{split}] Backup non trovato: {os.path.basename(backup)}")
    print()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostica e fix annotazioni border per Solar Panels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Comandi:
  (nessuno)   diagnosi (dry-run, nessuna modifica)
  diagnose    diagnosi esplicita
  fix         applica filtraggio border (crea backup automatico)
  restore     ripristina i JSON originali dal backup

Esempi:
  python fix_annotations.py
  python fix_annotations.py fix
  python fix_annotations.py fix --verbose
  python fix_annotations.py restore
        """
    )
    parser.add_argument("command", nargs="?", default="diagnose",
                        choices=["diagnose", "fix", "restore"],
                        help="Comando da eseguire (default: diagnose)")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostra singole annotazioni rimosse (solo con fix)")
    parser.add_argument("--train-json", type=str, default=TRAIN_JSON)
    parser.add_argument("--valid-json", type=str, default=VALID_JSON)
    args = parser.parse_args()



    if args.command in ("diagnose", None):
        cmd_diagnose(args)
    elif args.command == "fix":
        cmd_fix(args)
    elif args.command == "restore":
        cmd_restore(args)


if __name__ == "__main__":
    main()