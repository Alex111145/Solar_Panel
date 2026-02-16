#  Pipeline di Lavoro: Preparazione Dataset Pannelli Solari

Questa pipeline guida l'utente dal caricamento di un ortomosaico grezzo alla creazione di un dataset georiferito e ottimizzato per l'addestramento di modelli di Computer Vision.

---

## 📋 Requisiti Preliminari

* **File necessari:** Assicurati di avere `ortomosaico.tif` e `ortomosaico.tfw` nella stessa cartella degli script prima di iniziare.
* **Librerie Python:**
    ```bash
    pip install opencv-python simplekml
    ```

---

## 🚀 Fasi della Pipeline

### Step 1: Generazione delle Patch (Ritaglio)
**Script:** `step1_tile_generator.py`

Questo script rileva automaticamente l'ambiente (Locale o Server) e gestisce i percorsi in base alla macchina su cui lo esegui.

* **Cosa fa:** Carica l'ortomosaico e apre una finestra grafica per selezionare l'Area di Interesse (ROI).
* **Azione:** Disegna un rettangolo sui pannelli solari e premi **INVIO** o **SPAZIO**.
* **Risultato:** L'area viene tagliata in patch da **800x800 pixel** con un **overlap del 95%**. Le immagini vengono salvate in `my_thesis_data/`.

> **Risoluzione problemi SSH (Server Remoti):**
> Se ricevi l'errore `INTERFACCIA GRAFICA NON DISPONIBILE`, devi attivare il tunnel X11:
> 1. Scarica **VcXsrv** e configura **XLaunch** con "Disable access control" spuntato.
> 2. In **VS Code**, aggiungi `ForwardX11 yes` e `ForwardX11Trusted yes` nel tuo file config SSH.

---

### Step 2: Fase di Annotazione (Esterna)
*Questo passaggio avviene al di fuori di Python*.

* Carica le immagini di `my_thesis_data/` su una piattaforma come **Roboflow**.
* Esegui l'annotazione (creazione di box o poligoni sui pannelli).
* Esporta i dati in formato **COCO JSON** e salvali in: `datasets/solar_datasets`.

---

### Step 3: Creazione del Master JSON Globale
**Script:** `step2_global_coordinates.py`

Prende le annotazioni locali (0-800px) e le converte in coordinate globali riferite all'intero mosaico originale.

* **Dati richiesti:** Dovrai inserire le coordinate Pixel (X, Y) dell'ancora sul mosaico e le relative coordinate GPS (Lat/Lon).
* **Risultato:** Genera `master_global_anchored.json`, che contiene tutte le annotazioni georiferite.

---

### Step 4: Correzione e Rimappatura Classi
**Script:** `step3_util_remap_categories.py`

Utility per la pulizia e l'organizzazione delle categorie del dataset.

* **Eliminazione:** Se hai annotato una classe per errore, puoi eliminarla; lo script scalerà automaticamente gli ID delle altre classi.
* **Modifica ID:** Puoi cambiare manualmente l'ID di una classe per adattarlo ai requisiti di modelli specifici.
* **Sicurezza:** Crea automaticamente un file `_backup.json` prima di ogni modifica.

---
