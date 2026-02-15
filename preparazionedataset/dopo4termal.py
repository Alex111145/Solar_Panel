import json
import os

# CONFIGURAZIONE - Cambia i nomi dei file se necessario
FILES_TO_FIX = [
    "datasets/termal/train/_annotations.coco.json",
    "datasets/termal/valid/_annotations.coco.json"
]

def filter_json(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ File non trovato: {file_path}")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"\n📂 Analisi di: {file_path}")
    print(f"   - Annotazioni totali prima: {len(data['annotations'])}")

    # 1. Definiamo la nuova lista categorie (solo Thermal Fault)
    # Importante: le diamo l'ID 0 perché sarà l'unica classe
    new_categories = [{"id": 0, "name": "Thermal Fault", "supercategory": "none"}]
    
    # 2. Troviamo l'ID originale di "Thermal Fault" (che era 1 nel tuo caso)
    old_fault_id = None
    for cat in data['categories']:
        if cat['name'] == "Thermal Fault":
            old_fault_id = cat['id']
            break
    
    if old_fault_id is None:
        print(f"❌ Errore: Classe 'Thermal Fault' non trovata in questo file!")
        return

    # 3. Filtriamo le annotazioni: teniamo solo quelle di Thermal Fault e cambiamo ID in 0
    new_annotations = []
    for ann in data['annotations']:
        if ann['category_id'] == old_fault_id:
            ann['category_id'] = 0  # Diventa l'unica classe
            new_annotations.append(ann)

    # 4. Aggiorniamo il dizionario
    data['categories'] = new_categories
    data['annotations'] = new_annotations

    # Salva il nuovo file (sovrascrive o crea un backup)
    output_path = file_path.replace(".json", "_cleaned.json")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"✅ Completato!")
    print(f"   - Annotazioni rimaste (Thermal Fault): {len(new_annotations)}")
    print(f"   - Nuovo file salvato: {output_path}")

# Esecuzione
for path in FILES_TO_FIX:
    filter_json(path)