import os
import random
import json
import yaml
from PIL import Image
from tqdm import tqdm

def load_config(config_path="config.yaml"):
    """Legge il file di configurazione YAML."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_image_paths(folder_path):
    if not os.path.exists(folder_path):
        return []
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def create_crop_dataset_with_metadata(
    image_paths,
    output_dir,
    num_crops=1000,
    crop_size=128,
    block_size=256,
    seed=42
):
    if not image_paths:
        print(f"Nessuna immagine trovata per generare i crop in {output_dir}")
        return False

    input_dir = os.path.join(output_dir, "inputs")
    target_dir = os.path.join(output_dir, "targets")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    random.seed(seed)
    images = [(os.path.basename(p), Image.open(p).convert("RGB")) for p in image_paths]
    count = 0

    metadata = []

    with tqdm(total=num_crops, desc=f"Generating crops in {output_dir}") as pbar:
        while count < num_crops:
            img_name, img = random.choice(images)
            w, h = img.size

            if w < block_size or h < block_size:
                # Salta immagini troppo piccole
                continue

            # Coord blocco 256x256 nell'immagine originale
            x = random.randint(0, w - block_size)
            y = random.randint(0, h - block_size)
            target_block = img.crop((x, y, x + block_size, y + block_size))

            # Coord crop 128x128 nel blocco 256x256
            cx = random.randint(0, block_size - crop_size)
            cy = random.randint(0, block_size - crop_size)
            crop = target_block.crop((cx, cy, cx + crop_size, cy + crop_size))

            # Salva immagini
            filename = f"{count:05d}.png"
            crop.save(os.path.join(input_dir, filename))
            target_block.save(os.path.join(target_dir, filename))

            # Salva metadata
            metadata.append({
                "crop_filename": filename,
                "original_image": img_name,
                "block_coords": {"x": x, "y": y},
                "crop_coords_in_block": {"cx": cx, "cy": cy}
            })

            count += 1
            pbar.update(1)

    # Salva file JSON con metadati
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Dataset creato con successo in {output_dir} con {num_crops} campioni.")
    print(f"Metadata salvati in {metadata_path}")
    return True

def prepare_all_datasets():
    """Legge il config e prepara i dataset se non esistono."""
    config = load_config()
    
    # Parametri per il TRAINING
    train_sources = config['dataset']['train_sources']
    train_crops_folder = config['dataset']['train_crops_folder']
    train_num_samples = config['dataset']['train_num_samples']
    
    # Parametri per la VALUTAZIONE (Extraset)
    # Impostiamo di default 1000 crop per il test, se non è specificato nel config
    eval_sources = config['dataset']['eval_sources']
    eval_crops_folder = config['dataset']['eval_crops_folder']
    eval_num_samples = config['dataset'].get('eval_num_samples', 1000)

    # 1. Controllo Training Dataset
    train_inputs_path = os.path.join(train_crops_folder, "inputs")
    if not os.path.exists(train_inputs_path) or len(os.listdir(train_inputs_path)) < train_num_samples:
        print("\n[!] Dataset di Training mancante o incompleto. Generazione in corso...")
        train_images = get_image_paths(train_sources)
        if train_images:
            create_crop_dataset_with_metadata(train_images, train_crops_folder, num_crops=train_num_samples)
        else:
            print(f"ERRORE: Inserisci almeno un'immagine in {train_sources} per avviare il training.")
    else:
        print(f"[*] Dataset di Training già pronto in: {train_crops_folder}")

    # 2. Controllo Evaluation Dataset
    eval_inputs_path = os.path.join(eval_crops_folder, "inputs")
    if not os.path.exists(eval_inputs_path) or len(os.listdir(eval_inputs_path)) < eval_num_samples:
        print("\n[!] Dataset di Valutazione mancante o incompleto. Generazione in corso...")
        eval_images = get_image_paths(eval_sources)
        if eval_images:
            create_crop_dataset_with_metadata(eval_images, eval_crops_folder, num_crops=eval_num_samples)
        else:
            print(f"AVVISO: Nessuna immagine in {eval_sources}. I crop di valutazione non sono stati generati.")
    else:
        print(f"[*] Dataset di Valutazione già pronto in: {eval_crops_folder}")

if __name__ == "__main__":
    prepare_all_datasets()