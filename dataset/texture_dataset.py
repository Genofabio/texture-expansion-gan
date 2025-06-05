from torch.utils.data import Dataset 
from PIL import Image
import torchvision.transforms as T
import random
import os

class TextureFolderDataset(Dataset):
    def __init__(self, folder_path, num_samples=10000, transform=None):
        """
        folder_path: Cartella contenente le immagini da cui estrarre i blocchi.
        num_samples: Numero di campioni totali da generare.
        transform: Trasformazioni da applicare ai blocchi (es. ToTensor, Normalize).
        """
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.image_paths:
            raise ValueError(f"Nessuna immagine trovata in {folder_path}")
        
        self.images = [Image.open(p).convert('RGB') for p in self.image_paths]
        self.num_samples = num_samples
        self.k = 128
        self.transform = transform or T.ToTensor()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Scegli un'immagine a caso
        image = random.choice(self.images)
        width, height = image.size

        if width < 256 or height < 256:
            raise ValueError(f"L'immagine è troppo piccola: {width}x{height} - {image}")

        # Estrai un blocco 256×256
        x = random.randint(0, width - 256)
        y = random.randint(0, height - 256)
        target_block = image.crop((x, y, x + 256, y + 256))

        # Estrai un sotto-blocco 128×128 dal blocco target
        s_x = random.randint(0, 256 - 128)
        s_y = random.randint(0, 256 - 128)
        source_block = target_block.crop((s_x, s_y, s_x + 128, s_y + 128))

        # Trasformazioni
        source_tensor = self.transform(source_block)
        target_tensor = self.transform(target_block)

        return source_tensor, target_tensor, (s_x, s_y)
