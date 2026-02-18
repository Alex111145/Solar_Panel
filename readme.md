🚀 Pipeline di Lavoro: Preparazione Dataset Pannelli Solari
Questa pipeline guida l'utente dal caricamento di un ortomosaico grezzo alla creazione di un dataset georiferito e ottimizzato per l'addestramento di modelli di Computer Vision. Il sistema utilizza i dati GIS contenuti nel file World (.tfw) per garantire un ancoraggio geografico automatico e preciso.

📋 Requisiti Preliminari
File sorgente: Posiziona ortomosaico.tif e il relativo file ortomosaico.tfw nella cartella principale del progetto.

Librerie Python:

Bash
pip install opencv-python simplekml numpy folium pyperclip
🛠 Fasi della Pipeline
Step 1. Generazione delle Patch (Tiling)

Script: step1_tile_generator.py

Lo script gestisce l'estrazione delle immagini dal mosaico originale.

Selezione ROI (Area di Interesse): All'avvio si apre una preview dell'ortomosaico.

Azione: Disegna un rettangolo sull'area che contiene i pannelli solari e premi SPAZIO o INVIO.

Taglio Automatico: L'area selezionata viene tagliata in patch da 800x800 pixel con un overlap del 95%.

Filtraggio: Le patch che contengono troppi pixel vuoti (neri) vengono scartate per ottimizzare il dataset.

Output: Le immagini vengono salvate nella cartella my_thesis_data/. Ogni file include nel nome le proprie coordinate pixel (es. tile_col_X_row_Y.jpg).

Step 2. Annotazione (Esterna)

Passaggio manuale necessario per fornire la "conoscenza" al modello.

Carica le immagini di my_thesis_data/ su una piattaforma di annotazione (es. Roboflow).

Disegna i box (Bbox) o i poligoni attorno ai pannelli solari.

Esporta il dataset in formato COCO JSON.

Salva il file scaricato nel percorso: datasets/solar_datasets/train/_annotations.coco.json.

Step 3. Georeferenziazione Automatica

Script: step2_global_coordinates.py

Questo script agisce come ponte tra i pixel e le coordinate GPS reali.

Lettura TFW: Lo script legge automaticamente ortomosaico.tfw per ottenere la scala e l'origine geografica.

Calcolo GPS: * Converte le posizioni dei pannelli dai singoli file alle coordinate globali del mosaico.

Applica la trasformazione affine per ottenere Latitudine e Longitudine di ogni pannello.

Output: Genera il file finale master_global_anchored.json che contiene tutte le annotazioni arricchite con i dati GPS.

Step 4. Utility e Sanificazione

Script: step3_util_remap_categories.py

Utilizzato per pulire il dataset prima dell'addestramento.

Permette di eliminare classi errate o rimappare gli ID delle categorie.

Garantisce la coerenza tra i set di Train, Valid e Test.
