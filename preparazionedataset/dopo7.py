import json
import os

def clean_coco_report(json_path):
    if not os.path.exists(json_path):
        print(f"❌ Errore: Il file {json_path} non esiste.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    initial_count = len(data['annotations'])
    cleaned_ann = []
    
    # Contatori per il report
    stats = {
        "area_zero": 0,
        "width_zero": 0,
        "height_zero": 0,
        "invalid_coords": 0
    }

    for ann in data['annotations']:
        x, y, w, h = ann['bbox']
        
        # 1. Verifica Area
        if ann['area'] <= 0:
            stats["area_zero"] += 1
            continue
            
        # 2. Verifica Dimensioni BBox
        if w <= 0:
            stats["width_zero"] += 1
            continue
        if h <= 0:
            stats["height_zero"] += 1
            continue
            
        # 3. Verifica Coordinate (non devono essere negative, errore comune)
        if x < 0 or y < 0:
            stats["invalid_coords"] += 1
            continue
            
        cleaned_ann.append(ann)
    
    removed = initial_count - len(cleaned_ann)
    
    if removed > 0:
        data['annotations'] = cleaned_ann
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print("\n" + "!"*30)
        print(f"⚠️  MODIFICHE APPLICATE A: {os.path.basename(json_path)}")
        print(f"   Annotazioni totali rimosse: {removed}")
        if stats["area_zero"]: print(f"   - Maschere con area zero: {stats['area_zero']}")
        if stats["width_zero"] or stats["height_zero"]: print(f"   - BBox con dimensioni zero: {stats['width_zero'] + stats['height_zero']}")
        if stats["invalid_coords"]: print(f"   - Coordinate negative: {stats['invalid_coords']}")
        print("!"*30 + "\n")
    else:
        print(f"✅ {os.path.basename(json_path)} è già pulito. Nessuna modifica necessaria.")

# Configura i tuoi percorsi qui
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_NAME = "solar_datasets"
TRAIN_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train", "_annotations.coco.json")
VALID_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json")

print("🔍 Avvio scansione dataset...")
clean_coco_report(TRAIN_JSON)
clean_coco_report(VALID_JSON)