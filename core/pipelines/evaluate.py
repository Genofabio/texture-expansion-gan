import os
import json
import warnings
warnings.filterwarnings("ignore")
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import tqdm
import matplotlib
matplotlib.use('Agg') # Force headless mode to prevent memory leaks
import matplotlib.pyplot as plt

# Added for official perceptual metric calculation
import lpips 

from core.model.generator import Generator
from core.metrics.training_metrics import evaluate_training_batch_cached
from core.utils.file_utils import clear_directory

def load_config(config_path="config.yaml"):
    """Reads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

class PairedCropDataset(Dataset):
    def __init__(self, base_folder, transform=None):
        self.crops_folder = os.path.join(base_folder, 'inputs')
        self.targets_folder = os.path.join(base_folder, 'targets')
        
        if not os.path.exists(self.crops_folder):
            self.crop_filenames = []
        else:
            self.crop_filenames = sorted([
                f for f in os.listdir(self.crops_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.crop_filenames)

    def __getitem__(self, idx):
        crop_path = os.path.join(self.crops_folder, self.crop_filenames[idx])
        target_path = os.path.join(self.targets_folder, self.crop_filenames[idx])
        
        crop_img = Image.open(crop_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        
        if self.transform:
            crop_img = self.transform(crop_img)
            target_img = self.transform(target_img)
            
        return crop_img, target_img, self.crop_filenames[idx]

def denormalize(tensor):
    return (tensor + 1) / 2

def save_comparison_image(source, target, output, filename, save_dir):
    source_img = denormalize(source.detach().cpu()).clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    target_img = denormalize(target.detach().cpu()).clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    output_img = denormalize(output.detach().cpu()).clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Evaluation: {filename}", fontsize=16)

    axs[0].imshow(source_img)
    axs[0].set_title("Input 128x128")
    axs[0].axis('off')

    axs[1].imshow(target_img)
    axs[1].set_title("Target 256x256")
    axs[1].axis('off')

    axs[2].imshow(output_img)
    axs[2].set_title("Generated 256x256")
    axs[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"compare_{filename}")
    plt.savefig(save_path)
    
    # Aggressive cleanup to protect RAM
    fig.clf()
    plt.close('all')

def plot_metrics_distribution(detailed_results, save_dir):
    """Creates and saves metric histograms across the entire dataset."""
    psnr_vals = [r['psnr'] for r in detailed_results]
    ssim_vals = [r['ssim'] for r in detailed_results]
    lpips_vals = [r['lpips'] for r in detailed_results]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Evaluation Metrics Distribution across Dataset", fontsize=16)

    # PSNR (Higher is better)
    axs[0].hist(psnr_vals, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axs[0].set_title('PSNR Distribution')
    axs[0].set_xlabel('PSNR (dB)')
    axs[0].set_ylabel('Number of Images')
    axs[0].grid(axis='y', alpha=0.3)

    # SSIM (Higher is better, max 1.0)
    axs[1].hist(ssim_vals, bins=30, color='green', alpha=0.7, edgecolor='black')
    axs[1].set_title('SSIM Distribution')
    axs[1].set_xlabel('SSIM')
    axs[1].grid(axis='y', alpha=0.3)

    # LPIPS (Lower is better, min 0.0)
    axs[2].hist(lpips_vals, bins=30, color='red', alpha=0.7, edgecolor='black')
    axs[2].set_title('LPIPS Distribution')
    axs[2].set_xlabel('LPIPS')
    axs[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'metrics_distribution.png')
    plt.savefig(save_path)
    
    fig.clf()
    plt.close('all')
    print(f"✅ Distribution plot saved to: {save_path}")


def main():
    config = load_config()
    device_name = config['model']['device']
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    eval_crops_folder = config['dataset']['eval_crops_folder']
    metrics_output_dir = config['evaluation']['metrics_output']
    
    # Create output directories
    comparisons_output_dir = os.path.join(metrics_output_dir, 'comparisons')
    os.makedirs(metrics_output_dir, exist_ok=True)
    os.makedirs(comparisons_output_dir, exist_ok=True)

    # Clear old comparisons using our new utility
    clear_directory(comparisons_output_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = PairedCropDataset(eval_crops_folder, transform=transform)
    if len(dataset) == 0:
        print(f"\n[ERROR] No evaluation images found in {eval_crops_folder}.")
        return
        
    # BATCH SIZE 1: Crucial to avoid overloading the GPU during evaluation
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    weights_path = config['inference']['weights_path']
    if os.path.isdir(weights_path):
        checkpoint_files = sorted([f for f in os.listdir(weights_path) if f.endswith('.pt')])
        if checkpoint_files:
            weights_path = os.path.join(weights_path, checkpoint_files[-1])
        else:
            print(f"\n[ERROR] No valid .pt checkpoint files found in directory: {weights_path}")
            return

    print(f"Loading official model weights from: {weights_path}")
    
    model = Generator().to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint.get('generator_state_dict', checkpoint))
    model.eval()

    # --- INITIALIZE LPIPS HERE ---
    print("Initializing LPIPS perceptual model...")
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    detailed_results = []
    
    num_samples = len(dataset)

    # torch.no_grad() disables the training engine (zero gradients = zero VRAM wasted)
    with torch.no_grad():
        for i, (crop, target, filename) in enumerate(tqdm.tqdm(dataloader, desc="Generation and Evaluation")):
            crop = crop.to(device)
            target = target.to(device)
            
            # 1. Generate image
            generated = model(crop)

            # 2. Calculate metrics by explicitly passing the newly initialized LPIPS module
            metrics = evaluate_training_batch_cached(target, generated, device=device, lpips_module=lpips_fn)
            
            psnr_total += metrics['psnr']
            ssim_total += metrics['ssim']
            lpips_total += metrics['lpips']
            
            detailed_results.append({
                "filename": filename[0],
                "psnr": metrics['psnr'],
                "ssim": metrics['ssim'],
                "lpips": metrics['lpips']
            })

            # 3. Save comparison grid every 10 images
            if i % 10 == 0:
                save_comparison_image(crop, target, generated, filename[0], comparisons_output_dir)

    # Calculate Final Averages
    avg_psnr = psnr_total / num_samples
    avg_ssim = ssim_total / num_samples
    avg_lpips = lpips_total / num_samples

    aggregated_metrics = {
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "LPIPS": avg_lpips
    }

    print(f"\nSuccessfully processed {num_samples} evaluation samples.")
    print("--- AVERAGE METRICS RESULTS ---")
    for k, v in aggregated_metrics.items():
        print(f" - {k}: {v:.4f}")

    # Save JSON report
    final_output = {
        "aggregated_metrics": aggregated_metrics,
        "detailed_metrics": detailed_results
    }
    final_json_path = os.path.join(metrics_output_dir, "final_metrics.json")
    with open(final_json_path, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\n✅ Evaluation report saved to: {final_json_path}")
    print(f"✅ Comparison grids saved to: {comparisons_output_dir}")
    
    # Generate distribution plots
    plot_metrics_distribution(detailed_results, metrics_output_dir)

if __name__ == "__main__":
    main()