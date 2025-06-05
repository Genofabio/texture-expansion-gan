import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms

class StyleLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
        self.selected_layers = ['1', '6', '11', '20', '29']
        self.weights = [0.244, 0.061, 0.015, 0.004, 0.004]

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def gram_matrix(self, feature):
        (b, c, h, w) = feature.size()
        f = feature.view(b, c, h * w)
        G = torch.bmm(f, f.transpose(1, 2)) / (c * h * w)
        return G

    def forward(self, input, target):
        with torch.no_grad():
            input = self.transform(input)
            target = self.transform(target)

            style_loss = 0.0
            x = input
            y = target
            for i, layer in enumerate(self.vgg):
                x = layer(x)
                y = layer(y)
                if str(i) in self.selected_layers:
                    gm_x = self.gram_matrix(x)
                    gm_y = self.gram_matrix(y)
                    weight = self.weights[self.selected_layers.index(str(i))]
                    style_loss += weight * nn.functional.l1_loss(gm_x, gm_y)
        return style_loss
