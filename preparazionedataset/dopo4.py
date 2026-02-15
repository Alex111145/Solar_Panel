import json
import os

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_NAME = "solar_datasets"
TRAIN_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "train", "_annotations.coco.json")
VALID_JSON = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "valid", "_annotations.coco.json")

def check_coco_json(json_path, dataset_name):
    """
    Controlla il file JSON COCO e mostra le informazioni chiave
    """
    print(f"\n{'='*60}")
    print(f"🔍 ANALISI: {dataset_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(json_path):
        print(f"❌ File non trovato: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Informazioni base
    print(f"\n📊 Statistiche:")
    print(f"   - Immagini: {len(data.get('images', []))}")
    print(f"   - Annotazioni: {len(data.get('annotations', []))}")
    print(f"   - Categorie: {len(data.get('categories', []))}")
    
    # Categorie
    print(f"\n🏷️  Categorie trovate:")
    categories = data.get('categories', [])
    for cat in categories:
        print(f"   - ID: {cat.get('id')}, Nome: {cat.get('name')}")
    
    # Verifica duplicati
    category_names = [cat.get('name') for cat in categories]
    if len(category_names) != len(set(category_names)):
        print(f"\n⚠️  ATTENZIONE: Categorie duplicate trovate!")
        print(f"   Categorie: {category_names}")
    else:
        print(f"\n✅ Nessuna categoria duplicata")
    
    # Verifica annotazioni
    if len(data.get('annotations', [])) > 0:
        ann = data['annotations'][0]
        print(f"\n📝 Esempio annotazione:")
        print(f"   - Categoria ID: {ann.get('category_id')}")
        print(f"   - Ha bbox: {'bbox' in ann}")
        print(f"   - Ha segmentation: {'segmentation' in ann}")
        if 'segmentation' in ann:
            seg = ann['segmentation']
            if isinstance(seg, list):
                print(f"   - Tipo segmentation: polygon")
            elif isinstance(seg, dict):
                print(f"   - Tipo segmentation: RLE")
    
    print(f"\n{'='*60}\n")

# Controlla entrambi i file
print("\n🚀 CONTROLLO FILE ANNOTAZIONI COCO\n")
check_coco_json(TRAIN_JSON, "TRAIN")
check_coco_json(VALID_JSON, "VALID")

print("\n💡 SUGGERIMENTI:")
print("   1. Se vedi categorie duplicate, il file JSON ha un problema")
print("   2. Assicurati che ogni categoria abbia un ID univoco")
print("   3. Il campo 'name' deve essere identico tra train e valid")
print("   4. Se hai un solo oggetto da rilevare, dovresti avere 1 sola categoria\n")