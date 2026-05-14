import torch
import torch.nn as nn
import torch.optim as optim
from model.style_loss import StyleLoss
from model.perceptual_loss import PerceptualLoss

class GANTrainer:
    def __init__(self, generator, discriminator, device='cuda'):
        self.device = device

        # Usa i modelli passati come argomenti
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        # Verifica se il discriminatore è multi-scala
        self.multi_scale = hasattr(discriminator, 'discriminators')
        print(f"Using multi-scale discriminator: {self.multi_scale}")

        # Ottimizzatori
        self.optim_G = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # Loss functions
        self.adv_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

        # StyleLoss
        self.style_loss = StyleLoss(device).to(device)

        # PerceptualLoss
        self.perceptual_loss = PerceptualLoss(device=device)

        # Hyperparametri
        self.lambda_l1 = 50
        self.lambda_style = 10
        self.lambda_perceptual = 1

    def train_step(self, real_128, real_256, coords, return_generated=False):
        batch_size = real_128.size(0)
        real_128 = real_128.to(self.device)
        real_256 = real_256.to(self.device)

        # Genera fake_256 subito
        fake_256 = self.generator(real_128)

        # === Patch localizzata da real_256 e fake_256 ===
        localized_real = torch.zeros((batch_size, 3, 128, 128), device=self.device)
        localized_fake = torch.zeros((batch_size, 3, 128, 128), device=self.device)

        sx_list, sy_list = coords 
        for i in range(batch_size):
            sx = sx_list[i].item()
            sy = sy_list[i].item()
            localized_real[i] = real_256[i, :, sy:sy+128, sx:sx+128]
            localized_fake[i] = fake_256[i, :, sy:sy+128, sx:sx+128]

        # === Train Discriminator ===
        self.optim_D.zero_grad()

        if self.multi_scale:
            pred_real = self.discriminator(real_256)
            pred_fake = self.discriminator(fake_256.detach())

            # Crea tensori validi e fake in base alle dimensioni degli output
            valid = [torch.ones_like(p) for p in pred_real]
            fake = [torch.zeros_like(p) for p in pred_fake]

            loss_D = 0
            real_acc_total = 0
            fake_acc_total = 0

            for pr, vr, pf, fk in zip(pred_real, valid, pred_fake, fake):
                loss_D += self.adv_loss(pr, vr) + self.adv_loss(pf, fk)

                with torch.no_grad():
                    real_acc_total += (pr > 0.5).float().mean().item()
                    fake_acc_total += (pf < 0.5).float().mean().item()

            loss_D /= len(pred_real)
            real_acc = real_acc_total / len(pred_real)
            fake_acc = fake_acc_total / len(pred_fake)

        else:
            valid = torch.ones((batch_size, 1, 14, 14), device=self.device)
            fake = torch.zeros((batch_size, 1, 14, 14), device=self.device)

            pred_real = self.discriminator(real_256)
            pred_fake = self.discriminator(fake_256.detach())

            with torch.no_grad():
                real_acc = (pred_real > 0.5).float().mean().item()
                fake_acc = (pred_fake < 0.5).float().mean().item()

            loss_D = self.adv_loss(pred_real, valid) + self.adv_loss(pred_fake, fake)

        loss_D.backward()
        self.optim_D.step()

        # === Train Generator ===
        self.optim_G.zero_grad()

        pred_fake = self.discriminator(fake_256)

        if self.multi_scale:
            loss_G_adv = 0
            for pf in pred_fake:
                loss_G_adv += self.adv_loss(pf, torch.ones_like(pf))
            loss_G_adv /= len(pred_fake)
        else:
            loss_G_adv = self.adv_loss(pred_fake, torch.ones((batch_size, 1, 14, 14), device=self.device))

        loss_L1 = self.l1_loss(localized_fake, localized_real)
        loss_style = self.style_loss(fake_256, real_256)
        loss_perceptual = self.perceptual_loss(fake_256, real_256)

        loss_G = loss_G_adv + self.lambda_l1 * loss_L1 + self.lambda_style * loss_style + self.lambda_perceptual * loss_perceptual
        loss_G.backward()
        self.optim_G.step()

        training_info = {
            'loss_D': loss_D.item(),
            'loss_G': loss_G.item(),
            'loss_G_adv': loss_G_adv.item(),
            'loss_L1': loss_L1.item(),
            'loss_style': loss_style.item(),
            'loss_perceptual': loss_perceptual.item(),
            'acc_real': real_acc * 100,  
            'acc_fake': fake_acc * 100
        }

        if return_generated:
            return training_info, fake_256
        else:
            return training_info
