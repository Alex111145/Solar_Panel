#!/usr/bin/env python
"""
diagnose_annotations.py - Diagnostica Qualità Annotazioni COCO
===============================================================
Analizza i file JSON del dataset e stampa un report dettagliato su:
  - Numero di punti per poligono (4 punti = rettangolo grezzo, >12 = preciso)
  - Distribuzione delle aree (quanti pannelli sono small/medium/large)
  - Rapporto larghezza/altezza delle bounding box (aspect ratio)
  - Immagini con annotazioni sospette (poligoni troppo piccoli, sovrapposti, etc.)
  - Stima dell'impatto sulla metrica AP75

USO:
  python diagnose_annotations.py
  python diagnose_annotations.py --train-json /path/to/train.json
"""

import os
import sys
import json
import argparse
import math
from collections import Counter, defaultdict

# ==============================================================================
# CONFIGURAZIONE DEFAULT
# ==============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "solar_datasets")
TRAIN_JSON  = os.path.join(DATASET_DIR, "train", "_annotations.coco.json")
VALID_JSON  = os.path.join(DATASET_DIR, "valid", "_annotations.coco.json")

# ==============================================================================
# FUNZIONI DI ANALISI
# ==============================================================================

def load_json(path):
    if not os.path.exists(path):
        print(f"❌ File non trovato: {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)


def polygon_point_count(segmentation):
    """Restituisce il numero di punti nel poligono più grande."""
    if not segmentation or not isinstance(segmentation, list):
        return 0
    if isinstance(segmentation[0], list):
        # Formato lista di liste: [[x1,y1,x2,y2,...]]
        return max(len(seg) // 2 for seg in segmentation)
    return 0


def compute_iou_rect(box1, box2):
    """Calcola IoU tra due bbox in formato [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter = ix * iy
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0


def polygon_to_bbox_area(segmentation):
    """Calcola area del poligono con formula di Gauss (Shoelace)."""
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


def analyze_json(json_path, split_name):
    data = load_json(json_path)
    if not data:
        return

    anns   = data.get('annotations', [])
    images = data.get('images', [])

    print("\n" + "═" * 65)
    print(f"  📂 ANALISI: {split_name}")
    print(f"     File: {json_path}")
    print("═" * 65)

    if not anns:
        print("  ❌ Nessuna annotazione trovata.")
        return

    # --- Mappa immagini ---
    img_map = {img['id']: img for img in images}

    # --- Raccolta dati per ogni annotazione ---
    point_counts  = []
    areas         = []
    aspect_ratios = []
    poly_areas    = []
    suspicious    = []   # annotazioni potenzialmente problematiche
    per_image     = defaultdict(list)

    for ann in anns:
        img_id = ann.get('image_id')
        seg    = ann.get('segmentation', [])
        bbox   = ann.get('bbox', [0, 0, 0, 0])  # x, y, w, h
        area   = ann.get('area', 0)

        pts  = polygon_point_count(seg)
        point_counts.append(pts)
        areas.append(area)
        per_image[img_id].append(ann)

        w, h = bbox[2], bbox[3]
        if h > 0:
            aspect_ratios.append(w / h)

        # Area calcolata dal poligono (più precisa dell'area COCO)
        poly_area = polygon_to_bbox_area(seg)
        poly_areas.append(poly_area)

        # Annotazioni sospette
        issues = []
        if pts <= 4:
            issues.append(f"solo {pts} punti (rettangolo grezzo)")
        if area < 100:
            issues.append(f"area molto piccola ({area:.0f}px²)")
        if w > 0 and h > 0 and (w/h > 5 or h/w > 5):
            issues.append(f"aspect ratio estremo ({w/h:.1f})")
        if poly_area > 0 and area > 0:
            fill_ratio = min(poly_area, area) / max(poly_area, area)
            if fill_ratio < 0.7:
                issues.append(f"discrepanza area bbox/polygon ({fill_ratio:.2f})")

        if issues:
            img_name = img_map.get(img_id, {}).get('file_name', f'img_{img_id}')
            suspicious.append({
                'ann_id':   ann.get('id'),
                'img_name': img_name,
                'issues':   issues,
                'area':     area,
                'pts':      pts
            })

    total = len(anns)

    # ==========================================================================
    # REPORT 1: PUNTI PER POLIGONO
    # ==========================================================================
    print(f"\n📐 PUNTI PER POLIGONO (totale annotazioni: {total})")
    print("-" * 65)

    pt_counter = Counter(point_counts)
    buckets = {
        "4 punti (rettangolo grezzo — bassa precisione)":   [p for p in point_counts if p <= 4],
        "5-8 punti (poligono semplice — precisione media)": [p for p in point_counts if 5 <= p <= 8],
        "9-16 punti (poligono buono — buona precisione)":   [p for p in point_counts if 9 <= p <= 16],
        "17+ punti (poligono dettagliato — alta precisione)":[p for p in point_counts if p > 16],
    }

    for label, vals in buckets.items():
        pct = len(vals) / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {label}")
        print(f"    {len(vals):4d} ann  ({pct:5.1f}%)  {bar}")

    avg_pts = sum(point_counts) / len(point_counts)
    min_pts = min(point_counts)
    max_pts = max(point_counts)
    print(f"\n  Media punti/poligono : {avg_pts:.1f}")
    print(f"  Min / Max            : {min_pts} / {max_pts}")

    # Stima impatto su AP75
    pct_4pts = len(buckets["4 punti (rettangolo grezzo — bassa precisione)"]) / total * 100
    print(f"\n  💡 Impatto stimato su AP75:")
    if pct_4pts > 70:
        print(f"  🔴 {pct_4pts:.0f}% delle mask sono rettangoli a 4 punti.")
        print(f"     AP75 è strutturalmente limitato: il modello non può superare")
        print(f"     l'IoU del GT. Stima tetto AP75 ≈ {max(30, 55 - pct_4pts*0.2):.0f}%")
        print(f"     → AZIONE: ri-annotare con Smart Polygon almeno le 100 immagini")
        print(f"       più rappresentative del training set.")
    elif pct_4pts > 40:
        print(f"  🟡 {pct_4pts:.0f}% rettangoli grezzi. AP75 limitato ma non bloccato.")
        print(f"     → AZIONE: ri-annotare le immagini con >5 pannelli per frame.")
    else:
        print(f"  🟢 {pct_4pts:.0f}% rettangoli grezzi. Poligoni abbastanza precisi.")
        print(f"     Il problema AP75 è altrove (training, LR, etc.)")

    # ==========================================================================
    # REPORT 2: DISTRIBUZIONE AREE (COCO: S<32², M<96², L>96²)
    # ==========================================================================
    print(f"\n📏 DISTRIBUZIONE AREE (COCO: small<1024px², medium<9216px², large>9216px²)")
    print("-" * 65)

    small  = [a for a in areas if a < 32**2]
    medium = [a for a in areas if 32**2 <= a < 96**2]
    large  = [a for a in areas if a >= 96**2]

    for label, vals, tag in [
        ("Small  (<32²  =  1024 px²)", small,  "APs"),
        ("Medium (32²-96² = 1024-9216 px²)", medium, "APm"),
        ("Large  (>96²  =  9216 px²)", large,  "APl"),
    ]:
        pct = len(vals) / total * 100
        bar = "█" * int(pct / 2)
        avg = sum(vals)/len(vals) if vals else 0
        print(f"  {label}")
        print(f"    {len(vals):4d} ann  ({pct:5.1f}%)  area media: {avg:.0f}px²  {bar}")
        if tag == "APs" and len(vals) > 0:
            print(f"    ⚠️  Hai {len(vals)} oggetti small → se APs=0 il modello li perde tutti.")
        if tag == "APs" and len(vals) == 0:
            print(f"    ℹ️  Nessun oggetto small → APs=0.0 è normale e atteso.")

    # ==========================================================================
    # REPORT 3: ASPECT RATIO
    # ==========================================================================
    print(f"\n📐 ASPECT RATIO BOUNDING BOX")
    print("-" * 65)

    if aspect_ratios:
        avg_ar  = sum(aspect_ratios) / len(aspect_ratios)
        near_sq = sum(1 for ar in aspect_ratios if 0.7 <= ar <= 1.4)
        wide    = sum(1 for ar in aspect_ratios if ar > 1.4)
        tall    = sum(1 for ar in aspect_ratios if ar < 0.7)
        print(f"  Aspect ratio medio      : {avg_ar:.2f}  (1.0 = quadrato perfetto)")
        print(f"  Quasi quadrati (0.7-1.4): {near_sq:4d}  ({near_sq/total*100:.1f}%)")
        print(f"  Larghi (>1.4)           : {wide:4d}  ({wide/total*100:.1f}%)")
        print(f"  Alti (<0.7)             : {tall:4d}  ({tall/total*100:.1f}%)")

    # ==========================================================================
    # REPORT 4: DENSITÀ PER IMMAGINE
    # ==========================================================================
    print(f"\n🗂️  DENSITÀ ANNOTAZIONI PER IMMAGINE")
    print("-" * 65)

    counts_per_img = [len(v) for v in per_image.values()]
    if counts_per_img:
        avg_density = sum(counts_per_img) / len(counts_per_img)
        max_density = max(counts_per_img)
        dense_imgs  = sum(1 for c in counts_per_img if c > 10)
        print(f"  Immagini totali         : {len(per_image)}")
        print(f"  Pannelli medi/immagine  : {avg_density:.1f}")
        print(f"  Max pannelli in 1 img   : {max_density}")
        print(f"  Immagini con >10 pannelli: {dense_imgs}")
        if dense_imgs > 0:
            print(f"  💡 Le immagini dense sono le più importanti da ri-annotare")
            print(f"     con Smart Polygon (più pannelli = più impatto sulle metriche)")

    # ==========================================================================
    # REPORT 5: ANNOTAZIONI SOSPETTE
    # ==========================================================================
    print(f"\n⚠️  ANNOTAZIONI SOSPETTE ({len(suspicious)} su {total})")
    print("-" * 65)

    if not suspicious:
        print("  ✅ Nessuna annotazione sospetta trovata.")
    else:
        pct_susp = len(suspicious) / total * 100
        print(f"  {len(suspicious)} annotazioni ({pct_susp:.1f}%) con potenziali problemi:\n")
        # Mostra le prime 20, ordinate per numero di issues
        suspicious_sorted = sorted(suspicious, key=lambda x: len(x['issues']), reverse=True)
        for s in suspicious_sorted[:20]:
            issues_str = " | ".join(s['issues'])
            print(f"  ann_id {s['ann_id']:5d} | {s['img_name'][:40]:<40} → {issues_str}")
        if len(suspicious) > 20:
            print(f"\n  ... e altri {len(suspicious)-20} casi (vedi sopra per pattern generale)")

    # ==========================================================================
    # REPORT 6: RACCOMANDAZIONI FINALI
    # ==========================================================================
    print(f"\n🎯 RACCOMANDAZIONI PRIORITARIE")
    print("═" * 65)

    pct_4pts = len(buckets["4 punti (rettangolo grezzo — bassa precisione)"]) / total * 100
    pct_small = len(small) / total * 100

    rec_idx = 1
    if pct_4pts > 40:
        print(f"  {rec_idx}. 🔴 ALTA PRIORITÀ — Ri-annotare con Smart Polygon")
        print(f"     {pct_4pts:.0f}% delle mask sono rettangoli a 4 punti.")
        print(f"     Su Roboflow: seleziona le immagini → 'Smart Polygon' tool")
        print(f"     Obiettivo: portare i punti medi da {avg_pts:.0f} a >12 per poligono.")
        print(f"     Impatto atteso: +8-15% su AP75, +5-10% su AP complessivo.")
        rec_idx += 1

    if pct_small > 5:
        print(f"\n  {rec_idx}. 🟡 MEDIA PRIORITÀ — Oggetti small presenti ({pct_small:.0f}%)")
        print(f"     APs=0 indica che il modello li perde completamente.")
        print(f"     → Verifica che la risoluzione di training sia >= 800px short edge.")
        rec_idx += 1

    if len(per_image) < 50:
        print(f"\n  {rec_idx}. 🟡 MEDIA PRIORITÀ — Validation set piccolo ({len(per_image)} immagini)")
        print(f"     Con poche immagini le metriche hanno alta varianza.")
        print(f"     → Sposta alcune immagini da train a valid per arrivare a 80+.")
        rec_idx += 1

    print(f"\n  ℹ️  Il training v3 (MASK_FORMAT=polygon, MASK_WEIGHT=2.0) è già")
    print(f"     configurato per sfruttare al meglio i poligoni attuali.")
    print(f"     Ri-annotare + v3 = massimo impatto possibile senza cambiare backbone.")
    print("═" * 65 + "\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Diagnostica annotazioni COCO Solar Panels")
    parser.add_argument("--train-json", type=str, default=TRAIN_JSON)
    parser.add_argument("--valid-json", type=str, default=VALID_JSON)
    args = parser.parse_args()

    print("\n" + "═" * 65)
    print("  🔬 DIAGNOSTICA ANNOTAZIONI MASKDINO - SOLAR PANELS")
    print("═" * 65)

    analyze_json(args.train_json, "TRAINING SET")
    analyze_json(args.valid_json, "VALIDATION SET")


if __name__ == "__main__":
    main()
