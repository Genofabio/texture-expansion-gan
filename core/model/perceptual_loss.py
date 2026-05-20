import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        
        # Load pre-trained VGG16
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        # We only need the features up to the relu3_3 layer (index 16)
        self.vgg_layers = vgg[:16].to(device)
        self.vgg_layers.eval()
        
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        # FIX: Creazione dei tensori direttamente sulla GPU specificata
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def forward(self, input_img, target_img):
        # ImageNet normalization expects inputs in [0, 1], but our GAN outputs [-1, 1]
        input_img = (input_img + 1) / 2
        target_img = (target_img + 1) / 2

        # Apply normalization using the registered buffers
        input_img = (input_img - self.mean) / self.std
        target_img = (target_img - self.mean) / self.std

        # Extract features
        input_features = self.vgg_layers(input_img)
        target_features = self.vgg_layers(target_img)

        # Calculate Mean Squared Error (MSE) between feature representations
        return nn.functional.mse_loss(input_features, target_features)