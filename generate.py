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

# Importiamo dal nostro modulo core
from core.model.generator import Generator

def load_config(config_path="config.yaml"):
    """Legge il file di configurazione YAML."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    # 1. Caricamento config e setup device
    config = load_config()
    device_name = config['model']['device']
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Percorsi (Leggiamo tutto dal config!)
    # Assumiamo di aggiungere una sezione "generation" nel config, oppure usiamo fallback di default
    input_path = config.get('generation', {}).get('input_folder', './data/generation')
    output_path = config.get('generation', {}).get('output_folder', './outputs/generation')
    
    # Prende i pesi ufficiali esattamente come evaluate.py
    weights_path = config['inference']['weights_path']

    # Crea directory di output se non esiste
    os.makedirs(output_path, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"\n[ERRORE] Cartella di input non trovata: {input_path}")
        print("Crea la cartella e inserisci le immagini che vuoi elaborare.")
        return

    input_images = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not input_images:
        print(f"\n[AVVISO] Nessuna immagine trovata in {input_path}.")
        print("Inserisci le immagini da generare prima di avviare lo script.")
        return

    # 3. Caricamento Modello
    print(f"Caricamento pesi ufficiali da: {weights_path}")
    generator = Generator().to(device)
    
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        # Supporta sia il formato dizionario del training sia il tensore puro dei pesi
        generator.load_state_dict(checkpoint.get('generator_state_dict', checkpoint))
    except FileNotFoundError:
        print(f"\n[ERRORE] File dei pesi non trovato: {weights_path}.")
        print("Assicurati di aver inserito il tuo miglior checkpoint nella cartella 'weights/'")
        print("e che il percorso in config.yaml sia corretto.")
        return

    generator.eval()

    # 4. Pre-processing immagine di input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print(f"\nInizio generazione per {len(input_images)} immagini...")

    # 5. Generazione Immagini
    for img_name in input_images:
        full_input_path = os.path.join(input_path, img_name)
        
        try:
            img = Image.open(full_input_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Errore nel caricamento di {img_name}: {e}")
            continue

        start_time = time.time()

        with torch.no_grad():
            output = generator(img_tensor)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Post-processing e salvataggio
        output = (output + 1) / 2  # da [-1, 1] a [0, 1]
        
        # Salvataggio con un nome pulito che riprende l'originale
        out_name = f"{os.path.splitext(img_name)[0]}_generated.png"
        save_file_path = os.path.join(output_path, out_name)
        
        save_image(output, save_file_path)

        print(f"✅ '{img_name}' -> Elaborata in {elapsed_time:.2f}s | Salvata come: {out_name}")

    print(f"\nGenerazione completata con successo! Trovi i risultati in: {output_path}")

if __name__ == "__main__":
    main()