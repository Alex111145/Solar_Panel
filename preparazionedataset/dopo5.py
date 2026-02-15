import json
import os
import shutil
from datetime import datetime

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_NAME = "solar_datasets"

# Percorsi per i tre set: Train, Valid e Test
PATHS = {
    "TRAIN": os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train", "_annotations.coco.json"),
    "VALID": os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json"),
    "TEST":  os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "test", "_annotations.coco.json")
}

def fix_duplicate_categories(json_path, dataset_name):
    """
    Rimuove le categorie duplicate dal file JSON COCO e rimappa le annotazioni
    """
    if not os.path.exists(json_path):
        print(f"\n[!] Saltato {dataset_name}: File non trovato in {json_path}")
        return

    print(f"\n🔧 Fixing {dataset_name}: {os.path.basename(json_path)}")
    
    # Backup di sicurezza
    backup_path = json_path.replace('.json', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    shutil.copy2(json_path, backup_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    original_categories = data.get('categories', [])
    seen_names = {}
    id_mapping = {}
    new_categories = []
    new_id = 0
    
    for cat in original_categories:
        # Pulizia nome per evitare errori di battitura (es. "panel " vs "panel")
        name = cat['name'].strip()
        old_id = cat['id']
        
        # Usiamo il nome in minuscolo per il confronto tecnico
        lookup_name = name.lower()
        
        if lookup_name not in seen_names:
            seen_names[lookup_name] = new_id
            new_categories.append({
                'id': new_id,
                'name': name, # Manteniamo il nome originale per l'etichetta
                'supercategory': cat.get('supercategory', 'none')
            })
            id_mapping[old_id] = new_id
            new_id += 1
        else:
            id_mapping[old_id] = seen_names[lookup_name]
    
    # Aggiorna le annotazioni
    annotations = data.get('annotations', [])
    for ann in annotations:
        old_cat_id = ann['category_id']
        if old_cat_id in id_mapping:
            ann['category_id'] = id_mapping[old_cat_id]
    
    # Salva il file pulito
    data['categories'] = new_categories
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   ✅ {dataset_name} completato. Categorie ridotte da {len(original_categories)} a {len(new_categories)}.")

def main():
    print("\n" + "="*60)
    print("🛠️  FIX CATEGORIE DUPLICATE (TRAIN, VALID, TEST)")
    print("="*60)
    
    for name, path in PATHS.items():
        fix_duplicate_categories(path, name)
    
    print("\n" + "="*60)
    print("✅ TUTTI I SET SONO STATI CORRETTI!")
    print("="*60)

if __name__ == "__main__":
    main()