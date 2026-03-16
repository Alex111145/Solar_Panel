import os
import glob
import cv2
import torch
import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# ==============================================================
# CONFIGURAZIONE BASE
# ==============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER_NAME = "solar_datasets_nuoveannotazioni"   # <-- cambia se diverso
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output_inference")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "output_solar_swinLB", "model_final.pth")
CONFIG_FILE = os.path.join(BASE_DIR, "config.yaml")
SCORE_THRESHOLD = 0.5

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==============================================================
# FUNZIONE CONTEGGIO DATASET
# ==============================================================

def conta_immagini_dataset(base_dir, dataset_folder):
    splits = ["train", "valid", "test"]
    counts = {}

    for split in splits:
        path = os.path.join(base_dir, "datasets", dataset_folder, split)

        if not os.path.exists(path):
            counts[split] = 0
            continue

        images = (
            glob.glob(os.path.join(path, "*.jpg")) +
            glob.glob(os.path.join(path, "*.JPG")) +
            glob.glob(os.path.join(path, "*.png")) +
            glob.glob(os.path.join(path, "*.PNG"))
        )

        counts[split] = len(images)

    return counts

# ==============================================================
# CONFIGURAZIONE MODELLO
# ==============================================================

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_FILE)

    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg

# ==============================================================
# INFERENZA
# ==============================================================

def run_inference():

    print("="*60)
    print("   INFERENZA SOLAR PANELS — MaskDINO")
    print("="*60)

    # ----------------------------------------------------------
    # REPORT DATASET
    # ----------------------------------------------------------

    dataset_counts = conta_immagini_dataset(BASE_DIR, DATASET_FOLDER_NAME)

    print("\n📂 STRUTTURA DATASET")
    print("────────────────────────────────────────")
    print(f"  Train : {dataset_counts['train']} immagini")
    print(f"  Valid : {dataset_counts['valid']} immagini")
    print(f"  Test  : {dataset_counts['test']} immagini")

    print("\n📌 UTILIZZO DATASET")
    print("────────────────────────────────────────")
    print("  ➜ TRAINING      → cartella 'train'")
    print("  ➜ METRICHE AP   → cartella 'valid'")
    print("  ➜ INFERENZA     → cartella 'test'")

    # ----------------------------------------------------------
    # CARICAMENTO MODELLO
    # ----------------------------------------------------------

    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)

    # ----------------------------------------------------------
    # CARICAMENTO IMMAGINI TEST
    # ----------------------------------------------------------

    test_path = os.path.join(BASE_DIR, "datasets", DATASET_FOLDER_NAME, "test")

    image_paths = (
        glob.glob(os.path.join(test_path, "*.jpg")) +
        glob.glob(os.path.join(test_path, "*.JPG")) +
        glob.glob(os.path.join(test_path, "*.png")) +
        glob.glob(os.path.join(test_path, "*.PNG"))
    )

    if len(image_paths) == 0:
        print("\n❌ Nessuna immagine trovata nel test set.")
        return

    print(f"\n🚀 Avvio inferenza su {len(image_paths)} immagini...\n")

    # ----------------------------------------------------------
    # CICLO INFERENZA
    # ----------------------------------------------------------

    for img_path in image_paths:

        img = cv2.imread(img_path)
        outputs = predictor(img)

        instances = outputs["instances"].to("cpu")

        v = Visualizer(
            img[:, :, ::-1],
            MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            scale=1.0
        )

        out = v.draw_instance_predictions(instances)

        output_img = out.get_image()[:, :, ::-1]

        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_FOLDER, filename)

        cv2.imwrite(save_path, output_img)

        print(f"✔ Salvata: {filename}")

    print("\n✅ Inferenza completata.")
    print(f"📁 Risultati salvati in: {OUTPUT_FOLDER}")

# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    run_inference()