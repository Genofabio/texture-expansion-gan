import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import yaml
import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

# Import the model architecture from the core module
from core.model.generator import Generator

def load_config(config_path="config.yaml"):
    """Reads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    # 1. Load configuration and setup hardware device
    config = load_config()
    device_name = config['model']['device']
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Retrieve matched routing paths from config fields
    input_path = config['inference']['input_folder']
    output_path = config['inference']['output_folder']
    weights_path = config['inference']['weights_path']

    # Create destination directory if it does not exist
    os.makedirs(output_path, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"\n[ERROR] Input generation directory not found: {input_path}")
        print("Please create the directory and place the 128x128 textures you want to expand inside it.")
        return

    valid_extensions = ('.png', '.jpg', '.jpeg')
    input_images = [f for f in os.listdir(input_path) if f.lower().endswith(valid_extensions)]
    if not input_images:
        print(f"\n[WARNING] No valid images found in {input_path}.")
        print("Please insert the target source images before executing this script.")
        return

    # 3. Model Loading with automatic checkpoint selection
    if os.path.isdir(weights_path):
        checkpoint_files = sorted([f for f in os.listdir(weights_path) if f.endswith('.pt')])
        if checkpoint_files:
            weights_path = os.path.join(weights_path, checkpoint_files[-1])
        else:
            print(f"\n[ERROR] No valid .pt checkpoint files found in directory: {weights_path}")
            return

    print(f"Loading official model weights from: {weights_path}")
    generator = Generator().to(device)
    
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        # Supports both training dict checkpoint formats and raw weight tensors
        generator.load_state_dict(checkpoint.get('generator_state_dict', checkpoint))
    except FileNotFoundError:
        print(f"\n[ERROR] Weight file not found at: {weights_path}")
        print("Please ensure your training checkpoint directory or verify your config.yaml paths.")
        return

    generator.eval()

    # 4. Input image pre-processing configurations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print(f"\nStarting high-fidelity texture expansion for {len(input_images)} images...")

    # 5. Image Generation Loop
    # Move torch.no_grad outside the loop to further optimize performance
    with torch.no_grad():
        for img_name in input_images:
            full_input_path = os.path.join(input_path, img_name)
            
            try:
                img = Image.open(full_input_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
            except Exception as e:
                print(f"Error loading image asset {img_name}: {e}")
                continue

            start_time = time.time()

            # Autocast to save VRAM while maintaining speed
            with torch.amp.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu'):
                output = generator(img_tensor)

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Move output to CPU instantly to free up GPU memory
            output = output.cpu()

            # Post-processing map transformation: from [-1, 1] to [0, 1] range
            output = (output + 1) / 2
            
            # Save output using clean file naming conventions inherited from the source file
            out_name = f"{os.path.splitext(img_name)[0]}_generated.png"
            save_file_path = os.path.join(output_path, out_name)
            
            # Apply clamp to avoid visual glitches on extreme pixel values and save
            save_image(output.clamp(0, 1), save_file_path)

            print(f"✅ '{img_name}' -> Expanded in {elapsed_time:.2f}s | Saved as: {out_name}")

    print(f"\nGeneration tasks successfully completed! You can find the results inside: {output_path}")

if __name__ == "__main__":
    main()