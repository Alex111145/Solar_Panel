#  Pipeline di Lavoro: Preparazione Dataset Pannelli Solari

Questa pipeline guida l'utente dal caricamento di un ortomosaico grezzo alla creazione di un dataset georiferito, filtrato e ottimizzato per l'addestramento di modelli di Computer Vision.

---

## 📋 Requisiti Preliminari

* **File sorgente:** Posiziona `ortomosaico.tif` e il relativo file `ortomosaico.tfw` nella root del progetto.
* **Librerie Python:**
    ```bash
    pip install opencv-python simplekml numpy
    ```

---

##  Fasi della Pipeline

### Step 1. Generazione delle Patch e Punto di Ancora
**Script:** `step1_tile_generator.py`

> **⚠️ Alert** Su Linux/SSH, lo script verifica la presenza di X11. Se non rileva un display, si arresta consigliando come attivare x11 per il corretto funzionamento.

Questo script gestisce l'intera fase di estrazione dati e configurazione spaziale attraverso un'interfaccia grafica interattiva.

* **Rilevamento Percorsi:** Lo script identifica automaticamente se stai lavorando in locale (Windows/macOS) o su server, adattando le cartelle di input/output.
* **Selezione ROI (Area di Interesse):** * Si apre una preview dell'ortomosaico. 
    * **Azione:** Disegna un rettangolo sull'area dei pannelli e premi **SPAZIO** o **INVIO**.
* **Selezione Punto di Ancora:** * Dopo la ROI, lo script richiede di cliccare un punto specifico sull'immagine (es. l'angolo di un pannello o un punto di controllo GIS).
    * **Azione:** Clicca con il **tasto sinistro** del mouse. Le coordinate pixel reali (X, Y) vengono salvate automaticamente in `anchor_pixel_coords.json`.
* **Tiling e Filtraggio:** * L'area selezionata viene tagliata in patch da **800x800 pixel** con un **overlap del 95%**.
* **Output:** Immagini salvate in `my_thesis_data/`.

---

### Step 2. Annotazione (Esterna)
*Passaggio manuale su piattaforma cloud o locale.*

1.  Carica le patch generate su **Roboflow**
2.  Esegui l'annotazione (Bbox o Poligoni) sui pannelli solari.
3.  **Esporta** il dataset in formato **COCO JSON** e salvalo in: `datasets/solar_datasets/`.

---

### Step 3. Georeferenziazione Automatica
**Script:** `step2_global_coordinates.py`

> **⚠️ Alert** Su Linux/SSH, lo script verifica la presenza di X11. Se non rileva un display, si arresta consigliando come attivare x11 per il corretto funzionamento.


Questo script è il "ponte" tra il mondo dei pixel e le coordinate geografiche reali. Integra un sistema semi-automatico per evitare errori di trascrizione.

* **Caricamento Intelligente:** Recupera automaticamente le coordinate pixel dell'ancora salvate dallo Step 1.
* **Integrazione Browser (Maps Compact):** * Lo script apre automaticamente una finestra compatta di **Google Maps**.
    * **Azione:** Trova il punto corrispondente all'ancora sulla mappa, fai tasto destro e clicca sulle coordinate per copiarle.
* **Clipboard Monitoring (Auto-Paste):**
    * Non serve incollare nulla nel terminale! Lo script monitora gli appunti (clipboard). 
    * Appena rileva delle coordinate GPS valide, le acquisisce istantaneamente e procede con l'elaborazione.
* **Trasformazione Globale:** * Converte le Bounding Box locali delle patch in coordinate globali dell'intero mosaico.
    * Inserisce i metadati GPS nel file finale.
* **Output:** Genera `master_global_anchored.json`.

---
---

### Step 4. Utility: Gestione Categorie e Sanificazione
**Script:** `step3_util_remap_categories.py`

Questo script è fondamentale per preparare i file JSON all'addestramento, risolvendo conflitti di classi o errori di etichettatura tra i set di Train, Valid e Test.

* **Selezione Dataset Dinamica:** Lo script scansiona la cartella `datasets/` e ti permette di scegliere su quale progetto lavorare tramite un menu numerato.
* **Analisi Multi-Set:** Elabora contemporaneamente i file `train`, `valid` e `test`, garantendo che la mappatura delle classi sia coerente in tutto il dataset.
* **Due Modalità Operative:**
    * **[0] Eliminazione con Scalamento:** Rimuove una classe (es. un errore di annotazione) e trasla automaticamente gli ID delle altre classi verso il basso per eliminare i "buchi" (indispensabile per YOLO e Detectron2).
    * **[1] Rimappatura Manuale:** Permette di cambiare gli ID a piacimento (es. impostare "Pannello Integro" = 0, "Pannello Difettoso" = 1).
* **Filtro Annotazioni:** Quando una classe viene eliminata, lo script pulisce automaticamente tutte le annotazioni associate nel file JSON.
* **Sistema di Sicurezza:** Prima di ogni modifica, crea una copia di sicurezza `_annotations.coco_backup.json` per ogni set.

---
