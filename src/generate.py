import torch
from model.generator import Generator
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import os
import time
from datetime import datetime

def generate_image(input_path, weights_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carica modello
    generator = Generator().to(device)

    # Carica pesi
    checkpoint = torch.load(weights_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    # Pre-processing immagine di input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(input_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    start_time = time.time()

    # Generazione immagine
    with torch.no_grad():
        output = generator(img)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Post-processing e salvataggio
    output = (output + 1) / 2  # da [-1, 1] a [0, 1]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_image(output, output_path)

    print(f"Immagine generata in {elapsed_time:.2f} secondi")

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    generate_image(
        input_path='../data/generation/t.png',
        weights_path='../data/generation/weights_to_use.pt',
        output_path=f"../results/generation/output_{timestamp}.png"
    )