
Fase 1 — Taglia il nuovo ortomosaico in tile usando la stessa logica e lo stesso formato di nomi (tile_col_{X}_row_{Y}.jpg). Questo è fondamentale perché le coordinate X e Y nel nome del file sono l'offset in pixel che serve per ricostruire la posizione globale.
Fase 2 — Carica il tuo modello MaskDINO addestrato e lo fa girare su ogni tile. Le predizioni (bbox, maschere, score di confidenza) vengono salvate in un JSON con lo stesso identico formato COCO delle annotazioni manuali. In pratica il modello "annota" al posto tuo.
Fase 3 — Prende quel JSON e applica la trasformazione TFW (la stessa logica di convert_to_global.py) per calcolare latitudine e longitudine di ogni pannello trovato.
Per usarlo su un nuovo sito ti servono solo 3 cose: l'ortomosaico .tif, il suo file .tfw (che il software di fotogrammetria genera sempre insieme al mosaico), e i pesi del modello model_final.pth. Modifica i percorsi in cima allo script e lancialo.






Struttura

2 mosaici 
cartella con le cordinate
cartella con coco annodate


<img width="237" height="326" alt="image" src="https://github.com/user-attachments/assets/76cc449d-3ba1-4fc5-b364-788eda94fb1b" />





caricare detectron2


python -W ignore train_net.py

CUDA_VISIBLE_DEVICES=1 python -W ignore ultimotrain.py


python train_net.py --config-file tesi_config.yaml


Hardware Target: MacBook Air (CPU Optimization)

📝 Descrizione
Questo progetto implementa un modello di Segmentazione Semantica/Istanza (MaskDINO) per identificare e segmentare guasti sui pannelli solari.
Il codice è stato pesantemente personalizzato per funzionare in ambiente macOS (CPU), risolvendo incompatibilità native di CUDA, gestione della memoria e formattazione dei dataset COCO a classe singola.

🛠️ Installazione e Setup
Seguire questi passaggi per configurare l'ambiente da zero su macOS.

1. Creazione Environment

Bash
# Crea l'ambiente virtuale
python3 -m venv venv

# Attiva l'ambiente
source venv/bin/activate
2. Installazione Dipendenze

Bash
# Aggiorna pip e installa strumenti base
pip install --upgrade pip ninja

# Installa PyTorch (CPU version)
pip install torch torchvision

# Installa Detectron2 (da sorgente)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Installa librerie accessorie per MaskDINO
pip install opencv-python shapely timm scipy
📂 Struttura del Dataset
Il progetto si aspetta che i dati siano organizzati in questo modo nella root del progetto:

Plaintext
MaskDINO/
├── datasets/
│   └── tesi_dataset/
│       ├── train/              # Immagini di training (.jpg)
│       ├── val/                # Immagini di validazione (.jpg)
│       └── annotations/
│           ├── train.json      # Annotazioni COCO train
│           └── val.json        # Annotazioni COCO val
├── maskdino/                   # Codice sorgente modificato
├── output_tesi/                # Pesi (.pth) e log
├── train_net.py                # Script di addestramento
├── visualize_net.py            # Script di inferenza/demo
├── tesi_config.yaml            # Configurazione iperparametri
└── README.md
🚀 Come Eseguire
1. Avviare il Training



<img width="1470" height="956" alt="image" src="https://github.com/user-attachments/assets/298ace2f-1b90-4ab3-86b8-d2a837b04484" />




In Detectron2 e MaskDINO non si imposta il numero di "epoche" (quante volte il modello vede tutto il dataset), ma il numero di Iterazioni (quanti passi di aggiornamento fa il modello).

Ecco il calcolo esatto per la tua situazione:

1. Il Calcolo Matematico (Da mettere nella tesi!)

Dal tuo file di configurazione (tesi_config.yaml) e dai log, abbiamo questi dati:

Immagini di Training: 48 (dal tuo log Loaded 48 images).

Batch Size: 1 (abbiamo messo IMS_PER_BATCH: 1 nel file yaml per non fondere la CPU).

Iterazioni Totali: 2000 (impostato in MAX_ITER: 2000).

La formula è:

1 Epoca= 
Batch Size
Immagini Totali
​	
 = 
1
48
​	
 =48 iterazioni
Quindi, con 2000 iterazioni stai facendo:

48 iterazioni/epoca
2000 iterazioni
​	
 ≈41,6 Epoche
2. 2000 Iterazioni (41 Epoche) bastano?

Sì, assolutamente.
Per un dataset piccolo (48 immagini) e partendo da un modello pre-addestrato (Transfer Learning), 40-50 epoche sono il "punto giusto" (Sweet Spot).

Se ne fai troppe (es. 10.000 iterazioni): Il modello va in Overfitting. Impara a memoria le 48 foto e smette di riconoscere i guasti su foto nuove che non ha mai visto.

Se ne fai poche (es. 200 iterazioni): Il modello non capisce nulla (Underfitting).

3. Quanto tempo ci metterà?

Dai tuoi log vedo: time: 3.0326 secondi per iterazione.

2000 iterazioni×3 secondi=6000 secondi
6000/60=100 minuti≈1 ora e 40 minuti
È un tempo eccellente per un MacBook Air!

Il mio consiglio

Lascia finire le 2000 iterazioni previste.
Non interromperlo prima, a meno che tu non veda la total_loss scendere a zero spaccato (molto improbabile).

Cosa guardare:

Ora sei a loss 72.2.

Verso iterazione 500 dovrebbe essere scesa sotto 40.

Verso la fine (iter 1900) dovrebbe essere sotto 5-10.


