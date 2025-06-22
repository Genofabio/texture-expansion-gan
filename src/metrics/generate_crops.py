import os
import random
import json
from PIL import Image
from tqdm import tqdm

def get_image_paths(folder_path):
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

    print(f"Dataset creato in {output_dir} con {num_crops} campioni.")
    print(f"Metadata salvati in {metadata_path}")

if __name__ == "__main__":
    # Esempio d'uso:
    #folder_path_1img = "../data/training/images_model_1img"
    #folder_path_7img = "../data/training/images_model_7img"
    #folder_path_20img = "../data/training/images_model_20img"
    folder_extraset = "../data/evaluation/extraset/test_images"

    # Crea crops per modello 1 immagine
    # create_crop_dataset_with_metadata(
    #     get_image_paths(folder_path_1img),
    #     output_dir="../data/training/crops_model_1img",
    #     num_crops=1000
    # )

    # Crea crops per modello 7 immagini
    # create_crop_dataset_with_metadata(
    #     get_image_paths(folder_path_7img),
    #     output_dir="../data/training/crops_model_7img",
    #     num_crops=1000
    # )

    # Crea crops per modello 20 immagini
    # create_crop_dataset_with_metadata(
    #     get_image_paths(folder_path_20img),
    #     output_dir="../data/evaluation/intraset/model_20img",
    #     num_crops=1000
    # )

    # Crea crops per extraset
    create_crop_dataset_with_metadata(
        get_image_paths(folder_extraset),
        output_dir="../data/evaluation/extraset/",
        num_crops=1000
    )
