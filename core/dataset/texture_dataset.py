from torch.utils.data import Dataset 
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import os

class TextureFolderDataset(Dataset):
    def __init__(self, folder_path, num_samples=10000, transform=None, use_augmentation=False):
        """
        folder_path: Folder containing images from which to extract blocks.
        num_samples: Total number of samples to generate.
        transform: Transformations to apply to the blocks (e.g., ToTensor, Normalize).
        use_augmentation: If True, applies data augmentation on the blocks.
        """
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.image_paths:
            raise ValueError(f"No images found in {folder_path}")

        self.images = [Image.open(p).convert('RGB') for p in self.image_paths]
        self.num_samples = num_samples
        self.k = 128
        self.use_augmentation = use_augmentation

        if transform:
            self.transform = transform
        else:
            self.transform = None

    def __len__(self):
        return self.num_samples

    def apply_augmentation(self, image):
        # Synchronized data augmentation
        if self.use_augmentation:
            if random.random() < 0.5:
                image = TF.hflip(image)
            if random.random() < 0.5:
                image = TF.vflip(image)
            image = TF.adjust_brightness(image, 1 + (random.uniform(-0.2, 0.2)))
            image = TF.adjust_contrast(image, 1 + (random.uniform(-0.2, 0.2)))
            image = TF.adjust_saturation(image, 1 + (random.uniform(-0.2, 0.2)))
            image = TF.adjust_hue(image, random.uniform(-0.1, 0.1))
            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle)
        return image

    def __getitem__(self, idx):
        image = random.choice(self.images)
        width, height = image.size

        if width < 256 or height < 256:
            raise ValueError(f"Image is too small: {width}x{height} - {image}")

        x = random.randint(0, width - 256)
        y = random.randint(0, height - 256)
        target_block = image.crop((x, y, x + 256, y + 256))

        # Apply the same transformations to the target before extracting the source
        target_block = self.apply_augmentation(target_block)

        s_x = random.randint(0, 256 - 128)
        s_y = random.randint(0, 256 - 128)

        source_block = target_block.crop((s_x, s_y, s_x + 128, s_y + 128))

        # Apply ToTensor (and optional final transformation if provided)
        if self.transform:
            source_tensor = self.transform(source_block)
            target_tensor = self.transform(target_block)
        else:
            to_tensor = T.ToTensor()
            source_tensor = to_tensor(source_block)
            target_tensor = to_tensor(target_block)

        return source_tensor, target_tensor, (s_x, s_y)