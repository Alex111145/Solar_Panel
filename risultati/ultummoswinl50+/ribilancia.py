#!/usr/bin/env python
"""
rebalance_split.py - Sposta immagini da train a valid per stabilizzare le metriche
====================================================================================
PROBLEMA:
  Con solo 22 immagini in validation, AP/AP50/AP75 si appiattiscono.
  Il COCOEvaluator ha bisogno di almeno 80 immagini per metriche stabili.

USO:
  python rebalance_split.py                         # dry-run target=80
  python rebalance_split.py --apply                 # applica
  python rebalance_split.py --apply --target 60     # porta valid a 60 immagini
  python rebalance_split.py --apply --copy-files    # applica + copia jpg fisici
  python rebalance_split.py status                  # statistiche attuali
  python rebalance_split.py restore                 # ripristina backup
"""

import os
import sys
import json
import shutil
import random
import argparse
from collections import defaultdict

# ==============================================================================
# CONFIGURAZIONE DEFAULT
# ==============================================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "datasets", "solar_datasets_nuoveannotazioni")
DEFAULT_TRAIN_JSON = os.path.join(DATASET_DIR, "train", "_annotations.coco.json")
DEFAULT_VALID_JSON = os.path.join(DATASET_DIR, "valid", "_annotations.coco.json")
DEFAULT_TRAIN_IMG  = os.path.join(DATASET_DIR, "train")
DEFAULT_VALID_IMG  = os.path.join(DATASET_DIR, "valid")
DEFAULT_TARGET     = 80

# ==============================================================================
# FUNZIONI CORE
# ==============================================================================

def load_json(path):
    if not os.path.exists(path):
        print(f"❌ Non trovato: {path}")
        sys.exit(1)
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def do_backup(json_path):
    backup_path = json_path.replace(".json", "_backup_rebalance.json")
    if not os.path.exists(backup_path):
        shutil.copy2(json_path, backup_path)
        print(f"  💾 Backup: {os.path.basename(backup_path)}")
    else:
        print(f"  💾 Backup già esistente (non sovrascritto)")


def do_restore(json_path):
    backup_path = json_path.replace(".json", "_backup_rebalance.json")
    split = "TRAIN" if "train" in json_path else "VALID"
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, json_path)
        print(f"  ✅ [{split}] Ripristinato da backup")
    else:
        print(f"  ❌ [{split}] Backup non trovato: {os.path.basename(backup_path)}")


def get_img_stats(data):
    """Restituisce {img_id: {n_ann, file_name}} per ogni immagine."""
    stats = defaultdict(lambda: {'n_ann': 0, 'total_area': 0, 'file_name': ''})
    img_map = {img['id']: img.get('file_name', '') for img in data.get('images', [])}
    for ann in data.get('annotations', []):
        iid = ann['image_id']
        stats[iid]['n_ann'] += 1
        stats[iid]['total_area'] += ann.get('area', 0)
        stats[iid]['file_name'] = img_map.get(iid, '')
    return stats


def select_images_to_move(train_data, n_to_move, seed=42):
    """
    Seleziona immagini dal train da spostare in valid.
    Preferisce immagini con densità vicina alla media (le più "tipiche").
    """
    random.seed(seed)
    stats = get_img_stats(train_data)

    if not stats:
        return [], stats

    densities  = [s['n_ann'] for s in stats.values()]
    avg_density = sum(densities) / len(densities)

    # Ordina per distanza dalla media — prime quelle più rappresentative
    img_ids = list(stats.keys())
    img_ids.sort(key=lambda iid: abs(stats[iid]['n_ann'] - avg_density))

    # Shuffle leggero nella fascia candidati per varietà
    mid        = min(n_to_move * 3, len(img_ids))
    candidates = img_ids[:mid]
    random.shuffle(candidates)
    selected   = candidates[:n_to_move]

    return selected, stats


# ==============================================================================
# COMANDI
# ==============================================================================

def cmd_status(train_json, valid_json):
    print(f"\n{'═'*65}")
    print(f"  📊 STATO ATTUALE SPLIT")
    print(f"{'═'*65}")

    for json_path, split in [(train_json, "TRAIN"), (valid_json, "VALID")]:
        data  = load_json(json_path)
        imgs  = data.get('images', [])
        anns  = data.get('annotations', [])
        stats = get_img_stats(data)

        densities = [s['n_ann'] for s in stats.values()]
        avg_d = sum(densities)/len(densities) if densities else 0
        max_d = max(densities) if densities else 0

        areas  = [a.get('area', 0) for a in anns]
        small  = sum(1 for a in areas if a < 1024)
        medium = sum(1 for a in areas if 1024 <= a < 9216)
        large  = sum(1 for a in areas if a >= 9216)

        print(f"\n  [{split}]")
        print(f"    Immagini     : {len(imgs)}")
        print(f"    Annotazioni  : {len(anns)}")
        print(f"    Pannelli/img : {avg_d:.1f} media | {max_d} max")
        if areas:
            print(f"    Aree         : S={small} ({small/len(areas)*100:.0f}%)  "
                  f"M={medium} ({medium/len(areas)*100:.0f}%)  "
                  f"L={large} ({large/len(areas)*100:.0f}%)")

    print(f"\n{'═'*65}\n")


def cmd_rebalance(train_json, valid_json, train_img, valid_img,
                  target, apply, copy_files, seed):

    train_data = load_json(train_json)
    valid_data = load_json(valid_json)

    n_train = len(train_data.get('images', []))
    n_valid = len(valid_data.get('images', []))
    n_move  = max(0, target - n_valid)

    print(f"\n{'═'*65}")
    print(f"  📊 REBALANCE TRAIN/VALID SPLIT")
    print(f"{'═'*65}")
    print(f"  Train immagini attuali : {n_train}")
    print(f"  Valid immagini attuali : {n_valid}")
    print(f"  Target valid           : {target}")
    print(f"  Immagini da spostare   : {n_move}")
    if apply:
        print(f"  🔴 MODALITÀ APPLY")
    else:
        print(f"  🟡 DRY-RUN (aggiungi --apply per applicare)")

    if n_move <= 0:
        print(f"\n  ✅ Valid ha già {n_valid} immagini (>= {target}). Nulla da fare.")
        print(f"{'═'*65}\n")
        return

    if n_move > n_train * 0.3:
        print(f"\n  ⚠️  Stai spostando il {n_move/n_train*100:.0f}% del train.")
        print(f"     Considera --target 60 per non impoverire troppo il training.")

    # Selezione
    selected_ids, train_stats = select_images_to_move(train_data, n_move, seed)

    total_ann_moved = sum(train_stats[iid]['n_ann'] for iid in selected_ids)

    print(f"\n  Prime 10 immagini selezionate:")
    for iid in selected_ids[:10]:
        s = train_stats[iid]
        print(f"    {s['file_name'][:52]:<52} | {s['n_ann']} pannelli")
    if len(selected_ids) > 10:
        print(f"    ... e altre {len(selected_ids)-10} immagini")

    print(f"\n  Annotazioni spostate : {total_ann_moved}")
    print(f"  Train dopo           : {n_train - n_move} immagini")
    print(f"  Valid dopo           : {n_valid + n_move} immagini")

    if not apply:
        print(f"\n  ⚠️  DRY-RUN — nessuna modifica applicata.")
        print(f"{'═'*65}\n")
        return

    # === APPLICAZIONE ===
    selected_set = set(selected_ids)

    # Backup
    print(f"\n  Backup JSON...")
    do_backup(train_json)
    do_backup(valid_json)

    # Immagini e annotazioni da spostare
    imgs_to_move = [img for img in train_data['images'] if img['id'] in selected_set]
    anns_to_move = [ann for ann in train_data['annotations'] if ann['image_id'] in selected_set]

    # Train: rimuovi
    train_data['images']      = [img for img in train_data['images'] if img['id'] not in selected_set]
    train_data['annotations'] = [ann for ann in train_data['annotations'] if ann['image_id'] not in selected_set]

    # Valid: aggiungi con ID unici (evita conflitti)
    max_img_id = max((img['id'] for img in valid_data.get('images', [])), default=0)
    max_ann_id = max((ann['id'] for ann in valid_data.get('annotations', [])), default=0)

    id_remap   = {}
    new_img_id = max_img_id + 1
    for img in imgs_to_move:
        new_img             = dict(img)
        id_remap[img['id']] = new_img_id
        new_img['id']       = new_img_id
        valid_data['images'].append(new_img)
        new_img_id += 1

    new_ann_id = max_ann_id + 1
    for ann in anns_to_move:
        new_ann              = dict(ann)
        new_ann['id']        = new_ann_id
        new_ann['image_id']  = id_remap[ann['image_id']]
        valid_data['annotations'].append(new_ann)
        new_ann_id += 1

    # Salva JSON
    save_json(train_data, train_json)
    save_json(valid_data, valid_json)

    # Copia file fisici (opzionale)
    if copy_files:
        print(f"\n  📁 Copia file jpg fisici da train a valid...")
        copied, skipped, missing = 0, 0, 0
        for iid in selected_set:
            fname = train_stats[iid]['file_name']
            src   = os.path.join(train_img, fname)
            dst   = os.path.join(valid_img, fname)
            if os.path.exists(dst):
                skipped += 1
            elif os.path.exists(src):
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing += 1
        print(f"    Copiati: {copied} | Già esistenti: {skipped} | Non trovati: {missing}")

    print(f"\n  ✅ COMPLETATO:")
    print(f"     Train: {n_train} → {len(train_data['images'])} immagini")
    print(f"     Valid: {n_valid} → {len(valid_data['images'])} immagini")
    print(f"\n  ⚠️  Rilancia il training da zero dopo questo cambio:")
    print(f"     python train.py")
    print(f"{'═'*65}\n")


def cmd_restore(train_json, valid_json):
    print("\n🔄 Ripristino backup rebalance...")
    do_restore(train_json)
    do_restore(valid_json)
    print()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rebalance train/valid split per stabilizzare le metriche",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python rebalance_split.py                          # dry-run target=80
  python rebalance_split.py --apply                  # applica target=80
  python rebalance_split.py --apply --target 60      # applica target=60
  python rebalance_split.py --apply --copy-files     # applica + copia jpg
  python rebalance_split.py status                   # statistiche
  python rebalance_split.py restore                  # ripristina backup
        """
    )
    parser.add_argument("command", nargs="?", default="rebalance",
                        choices=["rebalance", "status", "restore"])
    parser.add_argument("--apply",      action="store_true")
    parser.add_argument("--target",     type=int, default=DEFAULT_TARGET)
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--train-json", type=str, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--valid-json", type=str, default=DEFAULT_VALID_JSON)
    args = parser.parse_args()

    train_json = args.train_json
    valid_json = args.valid_json

    if args.command == "status":
        cmd_status(train_json, valid_json)
    elif args.command == "restore":
        cmd_restore(train_json, valid_json)
    else:
        cmd_rebalance(
            train_json=train_json,
            valid_json=valid_json,
            train_img=DEFAULT_TRAIN_IMG,
            valid_img=DEFAULT_VALID_IMG,
            target=args.target,
            apply=args.apply,
            copy_files=args.copy_files,
            seed=args.seed
        )


if __name__ == "__main__":
    main()