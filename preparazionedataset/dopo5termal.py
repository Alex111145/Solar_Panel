import json
import os
import shutil

# --- CONFIGURAZIONE PERCORSI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Percorsi ai file originali che vuoi modificare
FILES_TO_FIX = [
    os.path.join(BASE_DIR, "datasets", "termal", "train", "_annotations.coco.json"),
    os.path.join(BASE_DIR, "datasets", "termal", "valid", "_annotations.coco.json")
]

def backup_and_fix(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ File non trovato: {file_path}")
        return

    # 1. CREAZIONE BACKUP
    backup_path = file_path.replace(".json", "_backup.json")
    shutil.copy2(file_path, backup_path)
    print(f"\n📦 Backup creato: {os.path.basename(backup_path)}")

    # 2. CARICAMENTO DATI
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 3. IDENTIFICAZIONE CLASSE 'Thermal Fault'
    # Cerchiamo l'ID originale indipendentemente da quale sia (0 o 1)
    old_fault_id = None
    for cat in data.get('categories', []):
        if cat['name'] == "Thermal Fault":
            old_fault_id = cat['id']
            break
    
    if old_fault_id is None:
        print(f"❌ Errore: Classe 'Thermal Fault' non trovata in {os.path.basename(file_path)}")
        return

    # 4. FILTRAGGIO CATEGORIE
    # Teniamo solo Thermal Fault e gli assegniamo ID 0
    data['categories'] = [{"id": 0, "name": "Thermal Fault", "supercategory": "none"}]

    # 5. FILTRAGGIO ANNOTAZIONI
    # Teniamo solo le annotazioni dei guasti e cambiamo l'ID a 0
    new_annotations = []
    for ann in data.get('annotations', []):
        if ann['category_id'] == old_fault_id:
            ann['category_id'] = 0
            new_annotations.append(ann)
    
    data['annotations'] = new_annotations

    # 6. SOVRASCRITTURA FILE ORIGINALE
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"✅ File sovrascritto: {os.path.basename(file_path)}")
    print(f"   - Rimaste {len(new_annotations)} annotazioni di guasti.")

# --- ESECUZIONE ---
header = "=" * 60
print(header)
print("🚀 INIZIO PULIZIA DATASET E CREAZIONE BACKUP")
print(header)

for path in FILES_TO_FIX:
    backup_and_fix(path)

print(f"\n{header}")
print("✨ OPERAZIONE COMPLETATA")
print("Ora puoi lanciare il tuo script di verifica per controllare i risultati.")
print(header)