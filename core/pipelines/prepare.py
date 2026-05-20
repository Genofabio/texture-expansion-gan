import os
import random
import json
import yaml
from PIL import Image
from tqdm import tqdm

def load_config(config_path="config.yaml"):
    """Reads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_image_paths(folder_path):
    """Retrieves image paths using the original basic directory listing logic."""
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
                continue

            x = random.randint(0, w - block_size)
            y = random.randint(0, h - block_size)
            target_block = img.crop((x, y, x + block_size, y + block_size))

            cx = random.randint(0, block_size - crop_size)
            cy = random.randint(0, block_size - crop_size)
            crop = target_block.crop((cx, cy, cx + crop_size, cy + crop_size))

            filename = f"{count:05d}.png"
            crop.save(os.path.join(input_dir, filename))
            target_block.save(os.path.join(target_dir, filename))

            metadata.append({
                "crop_filename": filename,
                "original_image": img_name,
                "block_coords": {"x": x, "y": y},
                "crop_coords_in_block": {"cx": cx, "cy": cy}
            })

            count += 1
            pbar.update(1)

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Dataset successfully created in {output_dir} with {num_crops} samples.")
    print(f"Metadata saved at {metadata_path}")

def prepare_all_datasets():
    """Reads configuration and prepares only the static evaluation dataset using original logic patterns."""
    config = load_config()
    
    eval_sources = config['dataset']['eval_sources']
    eval_crops_folder = config['dataset']['eval_crops_folder']
    eval_num_samples = config['dataset'].get('eval_num_samples', 1000)

    print("Training dataset is configured on-the-fly. Skipping training crop generation on disk.")

    eval_inputs_path = os.path.join(eval_crops_folder, "inputs")
    existing_crops = get_image_paths(eval_inputs_path)

    if len(existing_crops) < eval_num_samples:
        print("\n" + "="*50)
        print("Evaluation Dataset is missing or incomplete. Starting generation...")
        print("="*50 + "\n")
        
        eval_images = get_image_paths(eval_sources)
        if eval_images:
            create_crop_dataset_with_metadata(eval_images, eval_crops_folder, num_crops=eval_num_samples)
        else:
            print(f"WARNING: No images found in {eval_sources}. Evaluation crops were not generated.")
    else:
        print(f"Evaluation Dataset verified and ready in: {eval_crops_folder}")

if __name__ == "__main__":
    prepare_all_datasets()