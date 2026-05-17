import warnings
warnings.filterwarnings("ignore")
import yaml
from datetime import datetime
import os
import torch
import matplotlib.pyplot as plt
import time

# -- IMPORTAZIONE GENERATORE DATASET --
from prepare_dataset import prepare_all_datasets

from core.model.generator import Generator
from core.model.multi_scale_discriminator import MultiScaleDiscriminator
from core.model.gan import GANTrainer
from core.dataset.texture_dataset import TextureFolderDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from core.utils.visualize import visualize_generator_loss_components, visualize_sample
from core.utils.visualize import visualize_losses
from core.utils.init_weights import weights_init_normal
from core.utils.visualize import visualize_discriminator_accuracy
from core.utils.load_weights import load_weights

# -- AGGIORNATO: Importiamo con i nuovi nomi --
from core.metrics.training_metrics import evaluate_training_batch, save_training_metrics_plot
from core.utils.end_train import find_well_trained_step

def load_config(config_path="config.yaml"):
    """Legge il file di configurazione YAML."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    # 0. Prepara i dataset se non esistono!
    print("Verifica integrità dataset in corso...")
    prepare_all_datasets()
    
    # 1. Carica le impostazioni dal file config.yaml
    config = load_config()

    # 2. Estrai i percorsi e i parametri
    train_crops_folder = config['dataset']['train_crops_folder']
    checkpoints_output = config['training']['checkpoints_output']
    logs_output = config['training']['logs_output']
    
    num_samples = config['dataset']['train_num_samples']
    batch_size = config['training']['batch_size']
    num_steps = config['training']['num_steps']
    initial_lr = config['training']['initial_lr']
    device_name = config['model']['device']

    # 3. Configura il device
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. Crea le cartelle di output se non esistono già
    os.makedirs(checkpoints_output, exist_ok=True)
    os.makedirs(logs_output, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoints_output, f"weights_{timestamp}.pt")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Inizializza il dataset con i parametri del config
    dataset = TextureFolderDataset(folder_path=train_crops_folder, num_samples=num_samples, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # CORRETTO: Usiamo len(dataset) per evitare il doppio conteggio delle sottocartelle inputs/targets
    num_files = len(dataset)
    print(f"Numero totale di campioni univoci nel dataset di training: {num_files}")

    generator = Generator()
    discriminator = MultiScaleDiscriminator(num_scales=2)

    metrics_history = []
    
    # AGGIORNATO: Passiamo weights_dir corretto alla funzione
    start_step, losses, accuracies, metrics_history = load_weights(generator, discriminator, weights_dir=checkpoints_output)

    if start_step == 0:
        print("Inizializzazione manuale dei pesi (weights_init_normal)")
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    trainer = GANTrainer(generator, discriminator, device)

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

        # Linear decay learning rate (adattato per dinamicità se necessario)
        half_steps = num_steps // 2
        if step > half_steps:
            lr = initial_lr * (1 - (step - half_steps) / half_steps)
            for g in trainer.optim_G.param_groups: g['lr'] = lr
            for d in trainer.optim_D.param_groups: d['lr'] = lr

        if step % 100 == 0:
            losses, generated = trainer.train_step(s128, t256, coords, return_generated=True)
            end = time.time()

            # Passiamo esplicitamente logs_output alle funzioni di visualizzazione
            visualize_sample(s128, t256, generated, step=step, save_dir=logs_output)
            visualize_losses(losses_D_list, losses_G_list, timestamp, find_well_trained_step(losses_G_list, acc_real_list, acc_fake_list), save_dir=logs_output)
            visualize_generator_loss_components(losses_G_adv_list, losses_L1_list, losses_style_list, losses_perceptual_list, timestamp, save_dir=logs_output)
            visualize_discriminator_accuracy(acc_real_list, acc_fake_list, timestamp, save_dir=logs_output)

            # Valutazione Training metrics -------------------------------------
            metrics = evaluate_training_batch(t256, generated, device=device)
            if step == 0:
                metrics_history = []
            
            # ATTENZIONE: La chiave 'fid' è stata rimossa per allinearsi al nuovo dict
            metrics_history.append({
                'step': step,
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'lpips': metrics['lpips']
            })

            # Salva plot delle metriche ogni 100 step
            if step % 100 == 0 and len(metrics_history) > 1:
                save_training_metrics_plot(metrics_history, num_files, save_dir=logs_output)

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
        print(f"[{step}/{num_steps}] D: {losses['loss_D']:.4f} G: {losses['loss_G']:.4f} "
              f"(Adv: {losses['loss_G_adv']:.4f}, L1: {losses['loss_L1']:.4f}, Style: {losses['loss_style']:.4f}, Percep: {losses['loss_perceptual']:.4f}) "
              f"- {end - start:.3f}s")

        losses_D_list.append(losses['loss_D'])
        losses_G_list.append(losses['loss_G'])
        losses_G_adv_list.append(losses['loss_G_adv'])
        losses_L1_list.append(losses['loss_L1'])
        losses_style_list.append(losses['loss_style'])
        losses_perceptual_list.append(losses['loss_perceptual'])
        acc_real_list.append(losses['acc_real'])
        acc_fake_list.append(losses['acc_fake'])

    # Salvataggio finale
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