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

from core.model.generator import Generator
from core.metrics.training_metrics import evaluate_training_batch_cached

def load_config(config_path="config.yaml"):
    """Legge il file di configurazione YAML."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

class PairedCropDataset(Dataset):
    def __init__(self, base_folder, transform=None):
        """
        Si aspetta che la cartella contenga le sottocartelle 'inputs' e 'targets'
        generate dal nostro script prepare_dataset.py
        """
        self.crops_folder = os.path.join(base_folder, 'inputs')
        self.targets_folder = os.path.join(base_folder, 'targets')
        
        # Verifica che la cartella esista, altrimenti restituisce lista vuota
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

def evaluate_metrics_per_image(real_images, generated_images, device):
    results = []
    for real, gen in zip(real_images, generated_images):
        real_batch = real.unsqueeze(0)
        gen_batch = gen.unsqueeze(0)
        metrics = evaluate_training_batch_cached(real_batch, gen_batch, device=device)
        results.append(metrics)
    return results

def main():
    # 1. Caricamento config e device
    config = load_config()
    device_name = config['model']['device']
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Setup delle cartelle
    eval_crops_folder = config['dataset']['eval_crops_folder']
    metrics_output_dir = config['evaluation']['metrics_output']
    
    # AGGIORNATO: Impostato il nome definitivo in 'samples'
    samples_output_dir = os.path.join(metrics_output_dir, 'samples')
    
    os.makedirs(metrics_output_dir, exist_ok=True)
    os.makedirs(samples_output_dir, exist_ok=True)

    # 3. Setup Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = PairedCropDataset(eval_crops_folder, transform=transform)
    if len(dataset) == 0:
        print(f"\n[ERRORE] Nessuna immagine trovata in {eval_crops_folder}.")
        print("Assicurati di aver inserito le immagini in data/evaluation/originals e aver avviato prepare_dataset.py (o train.py).")
        return
        
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # 4. Caricamento Modello
    weights_path = config['inference']['weights_path']
    print(f"Caricamento pesi ufficiali da: {weights_path}")
    
    model = Generator().to(device)
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint.get('generator_state_dict', checkpoint))
    except FileNotFoundError:
        print(f"\n[ERRORE] File dei pesi non trovato: {weights_path}.")
        print("Assicurati di aver copiato il tuo miglior checkpoint nella cartella 'weights/'")
        print("oppure verifica che il percorso in config.yaml sia corretto.")
        return
        
    model.eval()

    generated_images = []
    real_images = []
    filenames_list = []

    # 5. Generazione Immagini (Inference)
    with torch.no_grad():
        for crops, targets, filenames in tqdm.tqdm(dataloader, desc="Generazione e Valutazione"):
            crops = crops.to(device)
            targets = targets.to(device)
            generated = model(crops)

            generated_images.append(generated.cpu())
            real_images.append(targets.cpu())
            filenames_list.extend(filenames)

            # Salva i PNG generati nella cartella samples
            for gen_img, fname in zip(generated, filenames):
                output_path = os.path.join(samples_output_dir, fname)
                gen_img = (gen_img + 1) / 2
                gen_img_pil = transforms.ToPILImage()(gen_img.clamp(0, 1))
                gen_img_pil.save(output_path)

    # 6. Calcolo Metriche
    generated_images = torch.cat(generated_images)
    real_images = torch.cat(real_images)

    print("Calcolo delle metriche (PSNR, SSIM, LPIPS) in corso...")
    detailed_metrics = evaluate_metrics_per_image(real_images, generated_images, device=device)

    detailed_results = []
    for fname, metric in zip(filenames_list, detailed_metrics):
        entry = {"filename": fname}
        entry.update(metric)
        detailed_results.append(entry)

    # Crea l'aggregato calcolando la media
    aggregated_metrics = {}
    for key in detailed_results[0]:
        if key == "filename":
            continue
        values = [r[key] for r in detailed_results if r[key] is not None]
        aggregated_metrics[key] = sum(values) / len(values) if values else 0

    print(f"\nElaborate {len(filenames_list)} immagini.")
    print("--- RISULTATI MEDI ---")
    for k, v in aggregated_metrics.items():
        print(f" - {k.upper()}: {v:.4f}")

    # 7. Salvataggio finale
    final_output = {
        "aggregated_metrics": aggregated_metrics,
        "detailed_metrics": detailed_results
    }

    final_json_path = os.path.join(metrics_output_dir, "final_metrics.json")
    with open(final_json_path, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\n✅ Report completo salvato in: {final_json_path}")

if __name__ == "__main__":
    main()