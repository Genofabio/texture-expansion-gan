import os
import json
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import tqdm

# Importa il modello generatore (modifica in base al tuo progetto)
from model.generator import Generator

from metrics.intraset_metrics import evaluate_intraset_cached

class PairedCropDataset(Dataset):
    def __init__(self, base_folder, transform=None):
        """
        base_folder deve contenere:
          - inputs/
          - targets/
          - generated/ (creata da noi)
        """
        self.crops_folder = os.path.join(base_folder, 'inputs')
        self.targets_folder = os.path.join(base_folder, 'targets')
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

def evaluate_intraset_per_image(real_images, generated_images, device):
    results = []
    for real, gen in zip(real_images, generated_images):
        real_batch = real.unsqueeze(0)
        gen_batch = gen.unsqueeze(0)
        metrics = evaluate_intraset_cached(real_batch, gen_batch, device=device)
        results.append(metrics)
    return results

def main(model_name, base_folder, weights_path, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_folder = os.path.join(base_folder, 'generated')
    os.makedirs(output_folder, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = PairedCropDataset(base_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    model = Generator().to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()

    generated_images = []
    real_images = []
    filenames_list = []

    with torch.no_grad():
        for crops, targets, filenames in tqdm.tqdm(dataloader, desc="Generazione immagini"):
            crops = crops.to(device)
            targets = targets.to(device)
            generated = model(crops)

            generated_images.append(generated.cpu())
            real_images.append(targets.cpu())
            filenames_list.extend(filenames)

            for gen_img, fname in zip(generated, filenames):
                output_path = os.path.join(output_folder, fname)
                if not os.path.exists(output_path):
                    gen_img = (gen_img + 1) / 2
                    gen_img_pil = transforms.ToPILImage()(gen_img.clamp(0, 1))
                    gen_img_pil.save(output_path)
                else:
                    pass

    generated_images = torch.cat(generated_images)
    real_images = torch.cat(real_images)

    detailed_metrics = evaluate_intraset_per_image(real_images, generated_images, device=device)

    detailed_results = []
    for fname, metric in zip(filenames_list, detailed_metrics):
        entry = {"filename": fname}
        entry.update(metric)
        detailed_results.append(entry)

    aggregated_metrics = {}
    for key in detailed_results[0]:
        if key == "filename":
            continue
        values = [r[key] if r[key] is not None else 0 for r in detailed_results]
        aggregated_metrics[key] = sum(values) / len(values)


    print(f"Elaborate {len(filenames_list)} immagini.")
    print(f"Metriche aggregate per {model_name}: {aggregated_metrics}")

    with open(os.path.join(output_folder, f"{model_name}_metrics_detailed.json"), 'w') as f:
        json.dump(detailed_results, f, indent=4)

    with open(os.path.join(output_folder, f"{model_name}_metrics_aggregated.json"), 'w') as f:
        json.dump(aggregated_metrics, f, indent=4)

if __name__ == "__main__":
    model_name = "model_7img"
    base_folder = "../data/evaluation/crops_model_7img"
    weights_path = "../data/evaluation/weights/Paper7Imm.pt"
    main(model_name, base_folder, weights_path)
