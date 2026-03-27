# 🛠️ Setup MaskDINO — Guida Installazione

## 🐍 1. Ambiente Virtuale Python

```bash
cd /root
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
```

---

## 🔥 2. Installazione PyTorch (CUDA 12.1)

Installa la versione di PyTorch compatibile con i driver del server (rilevato CUDA 12.4).

```bash
# Rimuovi versioni precedenti errate
pip uninstall torch torchvision torchaudio -y

# Installa la versione corretta per CUDA 12
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 👁️ 3. Installazione Detectron2

Installa Detectron2 dalla cartella locale disabilitando l'isolamento della build per permettere il riconoscimento di torch.

```bash
cd /root/detectron2
python -m pip install --no-build-isolation -e .
```

---

## 🧠 4. Compilazione Moduli CUDA MaskDINO

Compila l'operatore custom `MSDeformAttn` necessario per il decoder di MaskDINO.

```bash
# Spostati nella cartella degli operatori
cd /root/DinoMask/maskdino/modeling/pixel_decoder/ops

# Rendi lo script eseguibile ed avvialo
chmod +x make.sh
./make.sh
```

> **Nota:** Se `ninja` non è presente, installalo con:
> ```bash
> apt-get install ninja-build
> ```

---

## 📦 5. Dipendenze Progetto e Dataset

Installa le librerie per l'elaborazione dei dati e la gestione geospaziale.

```bash
cd /root
pip install opencv-python-headless numpy rasterio shapely pyyaml pandas
```

### Struttura Dataset

Assicurati che i dati siano organizzati come segue:

| Split      | Percorso                                                  |
|------------|-----------------------------------------------------------|
| Training   | `/root/datasets/solar_datasets/train/_annotations.coco.json` |
| Validation | `/root/datasets/solar_datasets/valid/_annotations.coco.json` |

---

## 🚀 6. Avvio dell'Addestramento (Multi-GPU)

Prima di avviare, configura il `PYTHONPATH` per includere i moduli di MaskDINO.

```bash
export PYTHONPATH=$PYTHONPATH:/root/DinoMask
```

Lancia il training su 2 GPU (es. ID 0 e 1):

```bash
CUDA_VISIBLE_DEVICES=0,1 python /root/train_net.py --num-gpus 2
```

---

## ✅ 7. Verifica Setup

Esegui lo script di controllo per validare l'installazione e l'integrità dei pesi.

```python
# Crea un file check_setup.py ed eseguilo
import torch
import os
import sys

sys.path.append("/root/DinoMask")

weights = "/root/DinoMask/weights/maskdino_swinl_pretrained.pth"
size_mb = os.path.getsize(weights) / (1024*1024) if os.path.exists(weights) else 0

print(f"CUDA: {torch.cuda.is_available()}")
print(f"Pesi Swin-L: {size_mb:.2f} MB (Target: >600MB)")

from maskdino.modeling.pixel_decoder.ops.modules import MSDeformAttn
print("✅ Moduli MaskDINO caricati!")
```
