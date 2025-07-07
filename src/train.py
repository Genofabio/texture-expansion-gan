import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import os
import torch
import matplotlib.pyplot as plt
import time
from model.generator import Generator
from model.multi_scale_discriminator import MultiScaleDiscriminator
from model.gan import GANTrainer
from dataset.texture_dataset import TextureFolderDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils.visualize import visualize_generator_loss_components, visualize_sample
from utils.visualize import visualize_losses
from utils.init_weights import weights_init_normal
from utils.visualize import visualize_discriminator_accuracy
from utils.load_weights import load_weights
from utils.count_tensors import count_tensors
from metrics.intraset_metrics import evaluate_intraset, save_intraset_metrics_plot
from utils.end_train import find_well_trained_step

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("../results/training/weights", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"../results/training/weights/weights_{timestamp}.pt"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    folder_path = '../data/training/images_model_7img/'
    num_files = sum([len(files) for _, _, files in os.walk(folder_path)])
    print(f"Numero totale di file nella cartella: {num_files}")

    dataset = TextureFolderDataset(folder_path='../data/training/images_model_7img/', num_samples=500000, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    generator = Generator()
    discriminator = MultiScaleDiscriminator(num_scales=2)


    metrics_history = []
    start_step, losses, accuracies, metrics_history = load_weights(generator, discriminator)

    if start_step == 0:
        print("Inizializzazione manuale dei pesi (weights_init_normal)")
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    trainer = GANTrainer(generator, discriminator, device)

    num_steps = 40000
    initial_lr = 2e-4

    losses_D_list = losses.get('losses_D_list', [])
    losses_G_list = losses.get('losses_G_list', [])
    losses_G_adv_list = losses.get('losses_G_adv_list', [])
    losses_L1_list = losses.get('losses_L1_list', [])
    losses_style_list = losses.get('losses_style_list', [])
    losses_perceptual_list = losses.get('losses_perceptual_list', [])
    acc_real_list = accuracies.get('acc_real_list', [])
    acc_fake_list = accuracies.get('acc_fake_list', [])

    print(f"\n-- Inizio train --\n")
    data_iter = iter(dataloader)
    for step in range(start_step, num_steps):
        start = time.time()

        try:
            s128, t256, coords = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            s128, t256, coords = next(data_iter)

        # Linear decay learning rate
        if step > 50000:
            lr = initial_lr * (1 - (step - 50000) / 50000)
            for g in trainer.optim_G.param_groups: g['lr'] = lr
            for d in trainer.optim_D.param_groups: d['lr'] = lr

        if step % 100 == 0:

            losses, generated = trainer.train_step(s128, t256, coords, return_generated=True)
            end = time.time()

            visualize_sample(s128, t256, generated, step=step)
            visualize_losses(losses_D_list, losses_G_list, timestamp, find_well_trained_step(losses_G_list, acc_real_list, acc_fake_list))
            visualize_generator_loss_components(losses_G_adv_list, losses_L1_list, losses_style_list, losses_perceptual_list, timestamp)
            visualize_discriminator_accuracy(acc_real_list, acc_fake_list, timestamp)

            # Valutazione Intraset metrics -------------------------------------
            metrics = evaluate_intraset(t256, generated, device=device)
            if step == 0:
                metrics_history = []
            metrics_history.append({
                'step': step,
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'lpips': metrics['lpips'],
                'fid': metrics['fid']
            })

            # Salva plot delle metriche ogni 1000 step
            if step % 100 == 0 and len(metrics_history) > 1:
                save_intraset_metrics_plot(metrics_history, num_files)

            # ------------------------------------------------------------------

            torch.save({
                'step': step,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': trainer.optim_G.state_dict(),
                'optimizer_D_state_dict': trainer.optim_D.state_dict(),
                'losses_D_list': losses_D_list,
                'losses_G_list': losses_G_list,
                'losses_G_adv_list': losses_G_adv_list,
                'losses_L1_list': losses_L1_list,
                'losses_style_list': losses_style_list,
                'losses_perceptual_list': losses_perceptual_list,
                'acc_real_list': acc_real_list,
                'acc_fake_list': acc_fake_list,
                'metrics_history': metrics_history
            }, checkpoint_path)

        else:
            losses = trainer.train_step(s128, t256, coords)
            end = time.time()

        # Visualizza informazioni
        cpu_tensors, gpu_tensors = count_tensors()
        print(f"[{step}/{num_steps}] D: {losses['loss_D']:.4f} G: {losses['loss_G']:.4f} "
              f"(Adv: {losses['loss_G_adv']:.4f}, L1: {losses['loss_L1']:.4f}, Style: {losses['loss_style']:.4f}, Percep: {losses['loss_perceptual']:.4f}) "
              f"- {end - start:.3f}s | CPU tensors: {cpu_tensors}, GPU tensors: {gpu_tensors}")

        losses_D_list.append(losses['loss_D'])
        losses_G_list.append(losses['loss_G'])
        losses_G_adv_list.append(losses['loss_G_adv'])
        losses_L1_list.append(losses['loss_L1'])
        losses_style_list.append(losses['loss_style'])
        losses_perceptual_list.append(losses['loss_perceptual'])
        acc_real_list.append(losses['acc_real'])
        acc_fake_list.append(losses['acc_fake'])

    torch.save({
        'step': num_steps,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': trainer.optim_G.state_dict(),
        'optimizer_D_state_dict': trainer.optim_D.state_dict(),
        'losses_D_list': losses_D_list,
        'losses_G_list': losses_G_list,
        'losses_G_adv_list': losses_G_adv_list,
        'losses_L1_list': losses_L1_list,
        'losses_style_list': losses_style_list,
        'losses_perceptual_list': losses_perceptual_list,
        'acc_real_list': acc_real_list,
        'acc_fake_list': acc_fake_list,
        'metrics_history': metrics_history
    }, checkpoint_path)


if __name__ == "__main__":
    main()