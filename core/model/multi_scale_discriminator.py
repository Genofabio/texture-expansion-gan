import torch.nn as nn
import torch.nn.functional as F
from core.model.discriminator import Discriminator  

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_scales=2):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([Discriminator() for _ in range(num_scales)])

    def forward(self, x):
        outputs = []
        for i in range(self.num_scales):
            if i > 0:
                x = F.avg_pool2d(x, kernel_size=2)
            outputs.append(self.discriminators[i](x))
        return outputs
