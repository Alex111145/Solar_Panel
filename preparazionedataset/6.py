import json
import os

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_NAME = "solar_datasets"
TRAIN_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train", "_annotations.coco.json")
VALID_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json")

def verify_json_ready(json_path, name):
    """Verifica se il file JSON è pronto per il training"""
    print(f"\n{'='*60}")
    print(f"🔍 VERIFICA: {name}")
    print(f"{'='*60}")
    
    if not os.path.exists(json_path):
        print(f"❌ File non trovato: {json_path}")
        return False
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    categories = data.get('categories', [])
    num_categories = len(categories)
    
    print(f"📊 Numero categorie: {num_categories}")
    
    # Mostra le categorie
    for cat in categories:
        print(f"   - ID: {cat['id']}, Nome: {cat['name']}")
    
    # Verifica duplicati
    names = [cat['name'] for cat in categories]
    unique_names = set(names)
    
    if len(names) != len(unique_names):
        print(f"\n❌ ERRORE: Categorie duplicate trovate!")
        print(f"   Categorie: {names}")
        print(f"   Univoche: {list(unique_names)}")
        return False
    
    # Verifica ID sequenziali (dovrebbero partire da 0)
    expected_ids = list(range(num_categories))
    actual_ids = sorted([cat['id'] for cat in categories])
    
    if actual_ids != expected_ids:
        print(f"\n⚠️  ATTENZIONE: Gli ID delle categorie non sono sequenziali!")
        print(f"   Attesi: {expected_ids}")
        print(f"   Trovati: {actual_ids}")
        print(f"   Questo potrebbe causare problemi...")
    
    # Verifica annotazioni
    annotations = data.get('annotations', [])
    if len(annotations) == 0:
        print(f"\n❌ ERRORE: Nessuna annotazione trovata!")
        return False
    
    # Verifica che tutte le annotazioni abbiano category_id validi
    invalid_categories = []
    for ann in annotations:
        cat_id = ann.get('category_id')
        if cat_id not in actual_ids:
            invalid_categories.append(cat_id)
    
    if invalid_categories:
        print(f"\n❌ ERRORE: Annotazioni con category_id non validi!")
        print(f"   IDs non validi trovati: {set(invalid_categories)}")
        print(f"   IDs validi: {actual_ids}")
        return False
    
    print(f"\n✅ File JSON OK!")
    print(f"   - {num_categories} categoria/e")
    print(f"   - {len(data.get('images', []))} immagini")
    print(f"   - {len(annotations)} annotazioni")
    
    return True

def main():
    print("\n" + "="*60)
    print("🛡️  VERIFICA PRE-TRAINING")
    print("="*60)
    
    train_ok = verify_json_ready(TRAIN_JSON, "TRAIN")
    valid_ok = verify_json_ready(VALID_JSON, "VALID")
    
    print(f"\n" + "="*60)
    print("📋 RIEPILOGO")
    print("="*60)
    
    if train_ok and valid_ok:
        # Verifica che le classi siano identiche
        with open(TRAIN_JSON, 'r') as f:
            train_data = json.load(f)
        with open(VALID_JSON, 'r') as f:
            valid_data = json.load(f)
        
        train_classes = sorted([cat['name'] for cat in train_data['categories']])
        valid_classes = sorted([cat['name'] for cat in valid_data['categories']])
        
        if train_classes == valid_classes:
            print(f"\n✅ TUTTO OK! Puoi procedere con il training!")
            print(f"\n🚀 Esegui: python train_net_v3.py")
            print(f"\n📊 Classi da rilevare: {train_classes}")
            return True
        else:
            print(f"\n❌ ERRORE: Le classi di train e valid non corrispondono!")
            print(f"   Train: {train_classes}")
            print(f"   Valid: {valid_classes}")
            print(f"\n🔧 Devi correggere i file JSON!")
            return False
    else:
        print(f"\n❌ CI SONO ERRORI NEI FILE JSON!")
        print(f"\n🔧 COSA FARE:")
        if not train_ok:
            print(f"   1. Il file TRAIN ha problemi")
        if not valid_ok:
            print(f"   2. Il file VALID ha problemi")
        print(f"\n💡 Esegui: python fix_duplicate_categories.py")
        return False

if __name__ == "__main__":
    success = main()
    print("\n")
    exit(0 if success else 1)