import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self, device, layers=['relu3_3', 'relu4_3']):
        super().__init__()
        weights = models.VGG19_Weights.DEFAULT
        vgg = models.vgg19(weights=weights).features.to(device).eval()
        self.blocks = nn.ModuleList()
        self.layer_names = layers
        self.layer_indices = {'relu1_1': 0, 'relu1_2': 2, 'relu2_1': 5,
                              'relu2_2': 7, 'relu3_1': 10, 'relu3_2': 12,
                              'relu3_3': 14, 'relu3_4': 16, 'relu4_1': 19,
                              'relu4_2': 21, 'relu4_3': 23}

        prev_index = 0
        for name in layers:
            index = self.layer_indices[name]
            block = nn.Sequential(*list(vgg.children())[prev_index:index+1])
            for param in block.parameters():
                param.requires_grad = False
            self.blocks.append(block)
            prev_index = index + 1

        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)

    def forward(self, input, target):
        with torch.no_grad():  # disabilita il tracking dei gradienti dentro questo blocco
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std
            for block in self.blocks:
                input = block(input)
                target = block(target)
        # ora input e target sono i tensori delle feature estratte da VGG, senza grafo
        # calcoliamo la loss normalmente (qui serve il grafo solo se vuoi retropropagare su input)
        loss = F.l1_loss(input, target)
        return loss

