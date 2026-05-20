import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")
import yaml
from datetime import datetime
import os
import torch
import time

from core.model.generator import Generator
from core.model.multi_scale_discriminator import MultiScaleDiscriminator
from core.model.gan import GANTrainer
from core.dataset.texture_dataset import TextureFolderDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from core.utils.visualize import visualize_generator_loss_components, visualize_sample
from core.utils.visualize import visualize_losses
from core.utils.init_weights import weights_init_normal
from core.utils.visualize import visualize_discriminator_accuracy
from core.utils.load_weights import load_weights
from core.metrics.training_metrics import evaluate_training_batch_cached, save_training_metrics_plot
from core.utils.end_train import find_well_trained_step

# Importiamo la funzione dal nuovo modulo di utility
from core.utils.file_utils import clear_directory

def load_config(config_path="config.yaml"):
    """Reads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    # 1. Load configuration settings from config.yaml
    config = load_config()

    # 2. Extract paths and structural hyperparameters
    train_sources = config['dataset']['train_sources']
    checkpoints_output = config['training']['checkpoints_output']
    logs_output = config['training']['logs_output']
    
    num_samples = config['dataset']['train_num_samples']
    batch_size = config['training']['batch_size']
    num_steps = config['training']['num_steps']
    initial_lr = config['training']['initial_lr']
    device_name = config['model']['device']

    # 3. Hardware device allocation
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. Create output directories if they do not exist
    os.makedirs(checkpoints_output, exist_ok=True)
    os.makedirs(logs_output, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoints_output, f"weights_{timestamp}.pt")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    num_files = sum([len(files) for _, _, files in os.walk(train_sources)])
    print(f"Total files in folder: {num_files}")

    dataset = TextureFolderDataset(folder_path=train_sources, num_samples=num_samples, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = Generator()
    discriminator = MultiScaleDiscriminator(num_scales=2)

    metrics_history = []
    
    # Load weights from checkpoints folder if available
    start_step, losses, accuracies, metrics_history = load_weights(generator, discriminator, weights_dir=checkpoints_output)

    if start_step == 0:
        print("Starting training from scratch: clearing old logs and checkpoints...")
        clear_directory(checkpoints_output)
        clear_directory(logs_output)
        
        # Le ricreiamo nel caso clear_directory abbia rimosso le cartelle radice
        os.makedirs(checkpoints_output, exist_ok=True)
        os.makedirs(logs_output, exist_ok=True)

        print("Manual weights initialization triggered (weights_init_normal)")
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
    else:
        print(f"Resuming training from step {start_step}. Preserving existing logs.")

    trainer = GANTrainer(generator, discriminator, device)

    losses_D_list = losses.get('losses_D_list', [])
    losses_G_list = losses.get('losses_G_list', [])
    losses_G_adv_list = losses.get('losses_G_adv_list', [])
    losses_L1_list = losses.get('losses_L1_list', [])
    losses_style_list = losses.get('losses_style_list', [])
    losses_perceptual_list = losses.get('losses_perceptual_list', [])
    acc_real_list = accuracies.get('acc_real_list', [])
    acc_fake_list = accuracies.get('acc_fake_list', [])

    print(f"\n-- Training Pipeline Started --\n")
    data_iter = iter(dataloader)
    
    for step in range(start_step, num_steps):
        start = time.time()

        try:
            s128, t256, coords = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            s128, t256, coords = next(data_iter)

        # Linear decay learning rate configuration matching the half-step decay logic
        half_steps = num_steps // 2
        if step > half_steps:
            lr = initial_lr * (1 - (step - half_steps) / half_steps)
            for g in trainer.optim_G.param_groups: g['lr'] = lr
            for d in trainer.optim_D.param_groups: d['lr'] = lr

        if step % 100 == 0:
            losses, generated = trainer.train_step(s128, t256, coords, return_generated=True)
            end = time.time()

            # Pass log directory explicitly to visualization functions
            visualize_sample(s128, t256, generated, step=step, save_dir=logs_output)
            visualize_losses(losses_D_list, losses_G_list, timestamp, find_well_trained_step(losses_G_list, acc_real_list, acc_fake_list), save_dir=logs_output)
            visualize_generator_loss_components(losses_G_adv_list, losses_L1_list, losses_style_list, losses_perceptual_list, timestamp, save_dir=logs_output)
            visualize_discriminator_accuracy(acc_real_list, acc_fake_list, timestamp, save_dir=logs_output)

            metrics = evaluate_training_batch_cached(t256, generated, device=device, lpips_module=trainer.perceptual_loss)
            if step == 0:
                metrics_history = []
            
            metrics_history.append({
                'step': step,
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'lpips': metrics['lpips']
            })

            # Save metrics plot tracking performance every 100 steps using physical file counters
            if step % 100 == 0 and len(metrics_history) > 1:
                save_training_metrics_plot(metrics_history, num_files, save_dir=logs_output)

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

        if step % 100 == 0 and step > 0:
            print("\n" + "="*50)
            print(f"EVALUATION METRICS UPDATE AT STEP {step}")
            print("="*50)

        # Clean performance log line without debug counters
        print(f"[{step}/{num_steps}] D: {losses['loss_D']:.4f} G: {losses['loss_G']:.4f} "
              f"(Adv: {losses['loss_G_adv']:.4f}, L1: {losses['loss_L1']:.4f}, Style: {losses['loss_style']:.4f}, Percep: {losses['loss_perceptual']:.4f}) "
              f"- {end - start:.3f}s")

        if step % 100 == 0 and step > 0:
            print("="*50 + "\n")

        losses_D_list.append(losses['loss_D'])
        losses_G_list.append(losses['loss_G'])
        losses_G_adv_list.append(losses['loss_G_adv'])
        losses_L1_list.append(losses['loss_L1'])
        losses_style_list.append(losses['loss_style'])
        losses_perceptual_list.append(losses['loss_perceptual'])
        acc_real_list.append(losses['acc_real'])
        acc_fake_list.append(losses['acc_fake'])

    # Final checkpoint dump at the end of training
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