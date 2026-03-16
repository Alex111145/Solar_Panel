#!/usr/bin/env python
"""
fix_annotations.py - Filtra annotazioni di pannelli tagliati ai bordi del tile
===============================================================================
PROBLEMA IDENTIFICATO:
  Il 52% delle annotazioni (5726/10923) ha aspect ratio estremo (~0.1) e
  discrepanza bbox/polygon ~0.50. Questi NON sono annotazioni sbagliate —
  sono pannelli reali ma TAGLIATI al bordo del tile durante la suddivisione
  del mosaico in patch.

  Effetto sul training:
  - Il modello impara che "pannello" può essere uno sliver di 10px
  - Le mask predette su pannelli interi non corrispondono
  - AP75 strutturalmente limitato perché IoU non può salire su oggetti troncati

  Effetto sul validation:
  - Il COCOEvaluator confronta mask predette (pannello intero) con GT troncato
  - IoU sempre bassa → AP50 ≈ AP75 ≈ AP

SOLUZIONE:
  Rimuovere dal JSON le annotazioni che rappresentano pannelli parziali,
  identificati da:
    1. Aspect ratio estremo (w/h < THRESH o h/w < THRESH) → pannello troncato
    2. Discrepanza bbox_area/polygon_area < FILL_THRESH → fill ratio basso

  Le annotazioni rimosse NON vengono cancellate dal filesystem —
  vengono solo escluse dal JSON di training/validation.
  Il JSON originale viene salvato come backup.

USO:
  python fix_annotations.py                    # dry-run: mostra solo cosa verrebbe rimosso
  python fix_annotations.py --apply            # applica le modifiche ai JSON
  python fix_annotations.py --apply --verbose  # applica e mostra ogni annotazione rimossa

PARAMETRI CHIAVE (modificabili):
  AR_THRESH   = 0.25  → rimuove se w/h < 0.25 o h/w < 0.25 (pannelli molto allungati)
  FILL_THRESH = 0.55  → rimuove se polygon_area/bbox_area < 0.55 (forma non rettangolare)
  MIN_AREA    = 200   → rimuove se area < 200px² (troppo piccolo per essere utile)
"""

import os
import sys
import json
import shutil
import argparse
import math
from collections import defaultdict

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "solar_datasets")
TRAIN_JSON  = os.path.join(DATASET_DIR, "train", "_annotations.coco.json")
VALID_JSON  = os.path.join(DATASET_DIR, "valid", "_annotations.coco.json")

# Soglie di filtraggio — modifica questi valori se vuoi essere più/meno aggressivo
AR_THRESH   = 0.25   # aspect ratio minimo accettabile (w/h e h/w)
FILL_THRESH = 0.55   # fill ratio minimo poligono/bbox
MIN_AREA    = 200    # area minima in px² (sotto = pannello troppo piccolo)

# ==============================================================================
# FUNZIONI
# ==============================================================================

def polygon_area(segmentation):
    """Area del poligono con formula di Gauss (Shoelace)."""
    if not segmentation or not isinstance(segmentation[0], list):
        return 0
    coords = segmentation[0]
    xs = coords[0::2]
    ys = coords[1::2]
    n  = len(xs)
    if n < 3:
        return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0


def is_border_panel(ann, ar_thresh, fill_thresh, min_area):
    """
    Restituisce (True, motivo) se l'annotazione è un pannello troncato al bordo.
    """
    bbox = ann.get('bbox', [0, 0, 1, 1])  # x, y, w, h
    area = ann.get('area', 0)
    seg  = ann.get('segmentation', [])

    w, h = bbox[2], bbox[3]
    if h <= 0 or w <= 0:
        return True, "bbox degenerata (w o h = 0)"

    # Filtro 1: area minima
    if area < min_area:
        return True, f"area troppo piccola ({area:.0f} < {min_area}px²)"

    # Filtro 2: aspect ratio estremo
    ar_wh = w / h
    ar_hw = h / w
    if ar_wh < ar_thresh:
        return True, f"pannello troncato: troppo alto ({w:.0f}x{h:.0f}, w/h={ar_wh:.2f})"
    if ar_hw < ar_thresh:
        return True, f"pannello troncato: troppo largo ({w:.0f}x{h:.0f}, h/w={ar_hw:.2f})"

    # Filtro 3: discrepanza poligono/bbox (fill ratio basso = forma non rettangolare)
    poly_a = polygon_area(seg)
    bbox_a = w * h
    if poly_a > 0 and bbox_a > 0:
        fill = min(poly_a, bbox_a) / max(poly_a, bbox_a)
        if fill < fill_thresh:
            return True, f"fill ratio basso ({fill:.2f} < {fill_thresh}) → pannello parziale"

    return False, ""


def filter_json(json_path, apply, verbose, ar_thresh, fill_thresh, min_area):
    """
    Analizza e (opzionalmente) filtra le annotazioni border da un JSON COCO.
    """
    if not os.path.exists(json_path):
        print(f"❌ File non trovato: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    anns         = data.get('annotations', [])
    total        = len(anns)
    to_remove    = []
    to_keep      = []
    reasons_count = defaultdict(int)

    for ann in anns:
        remove, reason = is_border_panel(ann, ar_thresh, fill_thresh, min_area)
        if remove:
            to_remove.append((ann, reason))
            # Categorizza il motivo principale
            if "troppo piccola" in reason:
                reasons_count["area piccola"] += 1
            elif "troncato" in reason:
                reasons_count["aspect ratio estremo"] += 1
            elif "fill ratio" in reason:
                reasons_count["fill ratio basso"] += 1
            else:
                reasons_count["altro"] += 1
        else:
            to_keep.append(ann)

    n_remove = len(to_remove)
    n_keep   = len(to_keep)
    pct      = n_remove / total * 100 if total > 0 else 0

    split_name = "TRAIN" if "train" in json_path else "VALID"
    print(f"\n{'═'*65}")
    print(f"  📂 {split_name}: {os.path.basename(json_path)}")
    print(f"{'═'*65}")
    print(f"  Annotazioni totali    : {total}")
    print(f"  Da rimuovere          : {n_remove} ({pct:.1f}%)")
    print(f"  Da mantenere          : {n_keep} ({100-pct:.1f}%)")

    print(f"\n  Motivi rimozione:")
    for reason, count in sorted(reasons_count.items(), key=lambda x: -x[1]):
        print(f"    {reason:<30}: {count:5d} ({count/total*100:.1f}%)")

    # Statistiche prima/dopo
    areas_keep   = [a['area'] for a in to_keep]
    areas_remove = [a[0]['area'] for a in to_remove]
    if areas_keep:
        print(f"\n  Area media (mantenute) : {sum(areas_keep)/len(areas_keep):.0f} px²")
    if areas_remove:
        print(f"  Area media (rimosse)   : {sum(areas_remove)/len(areas_remove):.0f} px²")

    if verbose and to_remove:
        print(f"\n  Prime 30 annotazioni rimosse:")
        img_map = {img['id']: img.get('file_name', '?') for img in data.get('images', [])}
        for ann, reason in to_remove[:30]:
            img_name = img_map.get(ann.get('image_id'), '?')
            print(f"    ann_id {ann['id']:6d} | {img_name[:35]:<35} | {reason}")

    if not apply:
        print(f"\n  ⚠️  DRY-RUN: nessuna modifica applicata.")
        print(f"     Rilancia con --apply per modificare i file.")
        return n_remove, n_keep

    # === APPLICAZIONE ===
    # Backup del file originale
    backup_path = json_path.replace(".json", "_backup_original.json")
    if not os.path.exists(backup_path):
        shutil.copy2(json_path, backup_path)
        print(f"\n  💾 Backup salvato: {os.path.basename(backup_path)}")
    else:
        print(f"\n  💾 Backup già esistente: {os.path.basename(backup_path)} (non sovrascritto)")

    # Aggiorna JSON
    data['annotations'] = to_keep

    # Rimuovi immagini che ora non hanno più annotazioni (opzionale ma pulito)
    img_ids_with_anns = set(a['image_id'] for a in to_keep)
    original_imgs     = data.get('images', [])
    filtered_imgs     = [img for img in original_imgs if img['id'] in img_ids_with_anns]
    n_imgs_removed    = len(original_imgs) - len(filtered_imgs)
    data['images']    = filtered_imgs

    with open(json_path, 'w') as f:
        json.dump(data, f)

    print(f"\n  ✅ JSON aggiornato:")
    print(f"     Annotazioni : {total} → {n_keep} (-{n_remove})")
    print(f"     Immagini    : {len(original_imgs)} → {len(filtered_imgs)} (-{n_imgs_removed} senza ann.)")

    return n_remove, n_keep


def restore_backup(json_path):
    """Ripristina il backup originale."""
    backup = json_path.replace(".json", "_backup_original.json")
    if os.path.exists(backup):
        shutil.copy2(backup, json_path)
        print(f"✅ Ripristinato: {json_path}")
    else:
        print(f"❌ Backup non trovato: {backup}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Filtra annotazioni di pannelli tagliati al bordo del tile"
    )
    parser.add_argument("--apply",       action="store_true",
                        help="Applica le modifiche ai JSON (default: dry-run)")
    parser.add_argument("--restore",     action="store_true",
                        help="Ripristina i backup originali")
    parser.add_argument("--verbose",     action="store_true",
                        help="Mostra le singole annotazioni rimosse")
    parser.add_argument("--ar-thresh",   type=float, default=AR_THRESH,
                        help=f"Soglia aspect ratio (default: {AR_THRESH})")
    parser.add_argument("--fill-thresh", type=float, default=FILL_THRESH,
                        help=f"Soglia fill ratio (default: {FILL_THRESH})")
    parser.add_argument("--min-area",    type=float, default=MIN_AREA,
                        help=f"Area minima px² (default: {MIN_AREA})")
    parser.add_argument("--train-json",  type=str, default=TRAIN_JSON)
    parser.add_argument("--valid-json",  type=str, default=VALID_JSON)
    args = parser.parse_args()

    print("\n" + "═"*65)
    print("  🔧 FIX ANNOTAZIONI — PANNELLI TAGLIATI AL BORDO TILE")
    print("═"*65)
    print(f"  Soglie applicate:")
    print(f"    Aspect ratio min  : {args.ar_thresh}  (w/h e h/w)")
    print(f"    Fill ratio min    : {args.fill_thresh} (polygon_area/bbox_area)")
    print(f"    Area minima       : {args.min_area} px²")
    if args.apply:
        print(f"  🔴 MODALITÀ APPLY: i file JSON verranno modificati!")
    else:
        print(f"  🟡 MODALITÀ DRY-RUN: nessuna modifica (aggiungi --apply per applicare)")

    if args.restore:
        print("\n🔄 Ripristino backup...")
        restore_backup(args.train_json)
        restore_backup(args.valid_json)
        return

    total_removed = 0
    total_kept    = 0

    for path in [args.train_json, args.valid_json]:
        r, k = filter_json(
            path,
            apply=args.apply,
            verbose=args.verbose,
            ar_thresh=args.ar_thresh,
            fill_thresh=args.fill_thresh,
            min_area=args.min_area
        )
        total_removed += r
        total_kept    += k

    print(f"\n{'═'*65}")
    print(f"  RIEPILOGO TOTALE")
    print(f"{'═'*65}")
    print(f"  Annotazioni rimosse  : {total_removed}")
    print(f"  Annotazioni mantenute: {total_kept}")
    print(f"  Riduzione dataset    : {total_removed/(total_removed+total_kept)*100:.1f}%")

    if not args.apply:
        print(f"\n  Prossimo passo → applica con:")
        print(f"  python fix_annotations.py --apply")
        print(f"\n  Poi rilancia il training v3 da zero:")
        print(f"  python train.py")
    else:
        print(f"\n  ✅ Fix applicata. Ora rilancia il training da zero:")
        print(f"     python train.py")
        print(f"\n  Per annullare tutto e tornare agli originali:")
        print(f"     python fix_annotations.py --restore")

    print("═"*65 + "\n")


if __name__ == "__main__":
    main()