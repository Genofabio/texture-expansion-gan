import matplotlib
matplotlib.use('Agg')  # Protezione memoria RAM
import os
import torch
import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure  
import matplotlib.pyplot as plt
from datetime import datetime

def save_training_metrics_plot(metrics_history, dataset_dim, save_dir='./outputs/training/logs'):
    """
    Saves a summary plot of the PSNR, SSIM, and LPIPS metrics monitored during training.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    steps = [m['step'] for m in metrics_history]
    psnr = [m['psnr'] for m in metrics_history]
    ssim = [m['ssim'] for m in metrics_history]
    lpips_vals = [m['lpips'] for m in metrics_history]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training Metrics Summary ({timestamp})", fontsize=18)

    # PSNR
    axs[0].plot(steps, psnr, color='blue')
    axs[0].set_title('PSNR')
    axs[0].set_xlabel('Step')
    axs[0].set_ylabel('PSNR (dB)')
    axs[0].grid(True)

    # SSIM
    axs[1].plot(steps, ssim, color='green')
    axs[1].set_title('SSIM')
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('SSIM')
    axs[1].grid(True)

    # LPIPS
    axs[2].plot(steps, lpips_vals, color='red')
    axs[2].set_title('LPIPS')
    axs[2].set_xlabel('Step')
    axs[2].set_ylabel('LPIPS (lower is better)')
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    save_path = os.path.join(plots_dir, f"training_metrics_dataset_dim_{dataset_dim}_{timestamp}.png")
    plt.savefig(save_path)
    
    # Pulizia memoria profonda
    fig.clf()
    plt.close('all')
    print(f"Saved Training metrics plot to: {save_path}")


def denormalize(tensor):
    return (tensor + 1) / 2  # [-1, 1] -> [0, 1]


def evaluate_training_batch(real_batch, generated_batch, device='cuda'):
    """Standard non-cached version used by the evaluation pipeline."""
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    loss_fn = lpips.LPIPS(net='vgg').to(device)

    psnr_scores, ssim_scores, lpips_scores = [], [], []

    with torch.no_grad():  
        for real_img, gen_img in zip(real_batch, generated_batch):
            real_img = denormalize(real_img).unsqueeze(0).to(device)
            gen_img = denormalize(gen_img).unsqueeze(0).to(device)

            psnr_scores.append(psnr(gen_img, real_img).item())
            ssim_scores.append(ssim(gen_img, real_img).item())
            lpips_scores.append(loss_fn(gen_img, real_img).item())

    avg_psnr = sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0.0
    avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0.0
    avg_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0.0

    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips
    }


def evaluate_training_batch_cached(real_batch, generated_batch, device='cuda', lpips_module=None):
    """Memory-optimized cached version used by the training pipeline (train.py)."""
    if not hasattr(evaluate_training_batch_cached, "initialized"):
        evaluate_training_batch_cached.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        evaluate_training_batch_cached.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        evaluate_training_batch_cached.initialized = True

    psnr = evaluate_training_batch_cached.psnr
    ssim = evaluate_training_batch_cached.ssim

    psnr_scores, ssim_scores = [], []

    with torch.no_grad():
        for real_img, gen_img in zip(real_batch, generated_batch):
            real_img = denormalize(real_img).unsqueeze(0).to(device)
            gen_img = denormalize(gen_img).unsqueeze(0).to(device)

            psnr_scores.append(psnr(gen_img, real_img).item())
            ssim_scores.append(ssim(gen_img, real_img).item())

    avg_psnr = sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0.0
    avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0.0

    if lpips_module is not None:
        with torch.no_grad():
            # FIX: Spostiamo esplicitamente entrambi i tensori sulla GPU prima di darli in pasto a LPIPS
            gen_gpu = generated_batch.to(device)
            real_gpu = real_batch.to(device)
            avg_lpips = lpips_module(gen_gpu, real_gpu).item()
    else:
        avg_lpips = 0.0

    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips
    }