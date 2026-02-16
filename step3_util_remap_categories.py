import json
import os
import shutil

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def get_dataset_selection(base_dir):
    datasets_root = os.path.join(base_dir, "datasets")
    if not os.path.exists(datasets_root):
        print(f"❌ Cartella 'datasets' non trovata in {base_dir}")
        exit()
    folders = [f for f in os.listdir(datasets_root) if os.path.isdir(os.path.join(datasets_root, f))]
    folders.sort()
    print("\n📂 DATASET DISPONIBILI:")
    for i, f in enumerate(folders):
        print(f"  [{i+1}] {f}")
    while True:
        try:
            choice = int(input("\n👉 Seleziona l'indice del dataset: ")) - 1
            if 0 <= choice < len(folders):
                return folders[choice]
        except ValueError: pass
        print("❌ Scelta non valida.")

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_name = get_dataset_selection(BASE_DIR)
    
    sets = ["train", "valid", "test"]
    file_paths = {}
    global_categories = {} # name -> id_originale
    
    # 1. Analisi preliminare
    for s in sets:
        path = os.path.join(BASE_DIR, "datasets", dataset_name, s, "_annotations.coco.json")
        if os.path.exists(path):
            file_paths[s] = path
            data = load_json(path)
            for cat in data.get('categories', []):
                global_categories[cat['name']] = cat['id']

    if not global_categories:
        print("❌ Nessun file COCO trovato nelle sottocartelle.")
        return

    # Ordinamento iniziale per ID
    sorted_cats = sorted(global_categories.items(), key=lambda x: x[1])
    cat_list = [item[0] for item in sorted_cats] 
    num_classi = len(cat_list)

    print("\n" + "="*40)
    print(f"📊 ANALISI DATASET: {dataset_name}")
    print("="*40)
    print(f"Trovate {num_classi} classi totali (ordinate per ID):")
    for i, name in enumerate(cat_list):
        print(f"  [{i}] {name} (ID attuale: {global_categories[name]})")
    print("="*40)

    print("\n🛠️  MENU OPERAZIONI")
    print("----------------------------------------")
    print("[0] ELIMINA una categoria (e scala gli ID verso l'alto)")
    print("[1] MODIFICA/RIMAPPA gli indici manualmente")
    
    while True:
        mode = input("\n👉 Scegli l'operazione (0 o 1): ").strip()
        if mode in ['0', '1']: break
        print("Scelta non valida.")

    new_mapping = {}
    to_delete = []

    # --- CASO 0: ELIMINAZIONE CON SCALAMENTO ---
    if mode == '0':
        # --- RE-VISUALIZZAZIONE MENU CLASSI ---
        print("\n" + "="*40)
        print(f"Trovate {num_classi} classi totali (ordinate per ID):")
        for i, name in enumerate(cat_list):
            print(f"  [{i}] {name} (ID attuale: {global_categories[name]})")
        print("="*40)

        while True:
            try:
                idx = int(input(f"\n👉 Inserisci l'indice (0-{num_classi-1}) della classe da CANCELLARE: "))
                if 0 <= idx < num_classi:
                    target_name = cat_list[idx]
                    to_delete.append(target_name)
                    print(f"\n⚠️  '{target_name}' verrà eliminata. Gli ID successivi scaleranno.")
                    
                    # Ricalcolo ID per le classi rimanenti
                    current_new_id = 0
                    print("\n--- ANTEPRIMA NUOVA MAPPATURA ---")
                    for name in cat_list:
                        if name != target_name:
                            new_mapping[name] = current_new_id
                            print(f"  {name}: ID {global_categories[name]} -> Nuovo ID {current_new_id}")
                            current_new_id += 1
                    print("---------------------------------")
                    break
            except ValueError: pass
            print("Indice non valido.")

    # --- CASO 1: MODIFICA MANUALE ---
    else:
        print("\n📝 ASSEGNAZIONE NUOVI INDICI:")
        for name in cat_list:
            while True:
                try:
                    new_id = int(input(f"  Nuovo ID per '{name}' (attuale {global_categories[name]}): "))
                    new_mapping[name] = new_id
                    break
                except ValueError: print("Inserisci un numero intero.")

    # 3. Applicazione modifiche
    confirm = input("\n⚠️ Procedere con la modifica dei file? (s/n): ").lower()
    if confirm != 's': return

    for s, path in file_paths.items():
        shutil.copy2(path, path.replace(".json", "_backup.json"))
        data = load_json(path)
        
        old_id_to_new = {}
        for cat in data.get('categories', []):
            name = cat['name']
            if name in to_delete:
                old_id_to_new[cat['id']] = "DELETE"
            else:
                old_id_to_new[cat['id']] = new_mapping.get(name)

        new_cats_json = []
        for name, n_id in new_mapping.items():
            new_cats_json.append({"id": n_id, "name": name, "supercategory": "none"})
        
        data['categories'] = sorted(new_cats_json, key=lambda x: x['id'])

        new_anns = []
        for ann in data.get('annotations', []):
            old_cid = ann['category_id']
            if old_cid in old_id_to_new:
                target = old_id_to_new[old_cid]
                if target != "DELETE" and target is not None:
                    ann['category_id'] = target
                    new_anns.append(ann)
        
        data['annotations'] = new_anns
        save_json(path, data)
        print(f"✅ {s.upper()}: Operazione completata. Nuove classi: {len(data['categories'])}")

    print("\n✨ Operazione completata. Gli ID sono stati scalati correttamente!")

if __name__ == "__main__":
    main()