
# Pipeline di Lavoro Preparazione Dataset

**Nota preliminare:** Assicurati di avere i file `ortomosaico.tif` e `ortomosaico.tfw` all'interno della stessa cartella degli script prima di iniziare.

---
### Step 1. Generazione delle Patch (Ritaglio)


installa pip install opencv-python
**Esegui lo script:** `step1_tile_generator`

Questo script è universale e gestisce automaticamente i percorsi in base alla macchina su cui lo esegui.

* **Cosa fa:** Carica l'ortomosaico (`ortomosaico.tif`) e apre una finestra grafica per selezionare l'Area di Interesse (ROI).
* **Azione:** Disegna un rettangolo sui pannelli solari e premi **INVIO** o **SPAZIO**.
* **Risultato:** L'area viene tagliata in patch da 800x800 pixel con un overlap del 95%. Le immagini vengono salvate in `my_thesis_data/`.

* **Casi d'uso e risoluzione problemi:**
1.  **Su PC Locale (Windows / Mac):** Lo script rileva il sistema operativo e apre direttamente la finestra di selezione.
2.  **Su Server Remoto (Lambda Labs via SSH):** * Se ricevi l'errore `INTERFACCIA GRAFICA NON DISPONIBILE`, devi attivare il tunnel X11.
    * **Scarica VcXsrv:** [https://sourceforge.net/projects/vcxsrv/](https://sourceforge.net/projects/vcxsrv/)
    * **Configura XLaunch:** Assicurati di spuntare **"Disable access control"** nelle impostazioni extra.
    * **Configura VS Code:** Aggiungi `ForwardX11 yes` e `ForwardX11Trusted yes` nel tuo file config SSH.
      
---

### Step su Roboflow. Fase di Annotazione
*Questo passaggio avviene al di fuori di Python.*

* Carica le immagini di `my_thesis_data/` su una piattaforma (es. Roboflow).
* Esegui l'annotazione (creazione di box o poligoni sui pannelli).
* Esporta i dati in formato **COCO JSON** e salvali nel percorso: `datasets/solar_datasets`.

---



### Step 2. Creazione del Master JSON Globale

installa pip install simplekml
**Esegui lo script:** `step2_global_coordinates.py`

* **Cosa fa:** Prende le annotazioni locali (0-800px) e le converte in coordinate globali riferite all'intero mosaico originale.
* **Dati richiesti:** Dovrai inserire le coordinate Pixel (X, Y) dell'ancora sul mosaico originale e le relative coordinate GPS.
* **Risultato:** Genera `master_global_anchored.json`, che contiene tutte le tue annotazioni georiferite.

---


### Step 3: Correzione e Rimappatura Classi
**Esegui lo script:** `step3_util_remap_categories.py`

* **Eliminazione:** Se hai annotato una classe per errore, puoi eliminarla; lo script scalerà automaticamente gli ID delle altre classi per non lasciare buchi.
* **Modifica ID:** Puoi cambiare manualmente l'ID di una classe se devi adattarlo ai requisiti di un modello specifico (es. far partire gli ID da 0).
* **Sicurezza:** Crea automaticamente un file `_backup.json` prima di ogni modifica.
