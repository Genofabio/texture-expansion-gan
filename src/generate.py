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

    # Crea directory di output se non esiste
    os.makedirs(output_path, exist_ok=True)

    input_images = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in input_images:
        full_input_path = os.path.join(input_path, img_name)
        img = Image.open(full_input_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        start_time = time.time()

        # Generazione immagine
        with torch.no_grad():
            output = generator(img_tensor)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Post-processing e salvataggio
        output = (output + 1) / 2  # da [-1, 1] a [0, 1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{os.path.splitext(img_name)[0]}_gen_{timestamp}.png"
        save_image(output, os.path.join(output_path, out_name))

        print(f"Immagine '{img_name}' generata in {elapsed_time:.2f} secondi")

if __name__ == "__main__":
    generate_image(
        input_path='../data/generation/',
        weights_path='../data/evaluation/extraset/model_20img_augmented/weights/Paper20Augmented.pt',
        output_path='../results/generation/'
    )
