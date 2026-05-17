import os
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure  
import lpips
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

def save_training_metrics_plot(metrics_history, dataset_dim, save_dir='./outputs/training/logs'):
    """
    Salva un grafico riassuntivo delle metriche PSNR, SSIM e LPIPS durante il training.
    metrics_history è una lista di dict, es:
    [{'step': 0, 'psnr': ..., 'ssim': ..., 'lpips': ...}, ...]
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    steps = [m['step'] for m in metrics_history]
    psnr = [m['psnr'] for m in metrics_history]
    ssim = [m['ssim'] for m in metrics_history]
    lpips = [m['lpips'] for m in metrics_history]

    # Grafico 1x3 per le tre metriche rimanenti
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
    axs[2].plot(steps, lpips, color='red')
    axs[2].set_title('LPIPS')
    axs[2].set_xlabel('Step')
    axs[2].set_ylabel('LPIPS (lower is better)')
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    save_path = os.path.join(plots_dir, f"training_metrics_dataset_dim_{dataset_dim}_{timestamp}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Salvato Training metrics plot in: {save_path}")

def denormalize(tensor):
    return (tensor + 1) / 2  # [-1, 1] -> [0, 1]

def evaluate_training_batch(real_batch, generated_batch, device='cuda'):
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    loss_fn = lpips.LPIPS(net='vgg').to(device)

    psnr_scores, ssim_scores, lpips_scores = [], [], []

    with torch.no_grad():  # 🔒 evita calcolo del grafo
        for real_img, gen_img in zip(real_batch, generated_batch):
            real_img = denormalize(real_img).unsqueeze(0).to(device)
            gen_img = denormalize(gen_img).unsqueeze(0).to(device)

            psnr_scores.append(psnr(gen_img, real_img).item())
            ssim_scores.append(ssim(gen_img, real_img).item())
            lpips_scores.append(loss_fn(gen_img, real_img).item())

    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    avg_lpips = sum(lpips_scores) / len(lpips_scores)

    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips
    }

def evaluate_training_batch_cached(real_batch, generated_batch, device='cuda'):
    if not hasattr(evaluate_training_batch_cached, "initialized"):
        evaluate_training_batch_cached.psnr = PeakSignalNoiseRatio().to(device)
        evaluate_training_batch_cached.ssim = StructuralSimilarityIndexMeasure().to(device)
        evaluate_training_batch_cached.loss_fn = lpips.LPIPS(net='vgg').to(device)
        evaluate_training_batch_cached.initialized = True

    psnr = evaluate_training_batch_cached.psnr
    ssim = evaluate_training_batch_cached.ssim
    loss_fn = evaluate_training_batch_cached.loss_fn

    psnr_scores, ssim_scores, lpips_scores = [], [], []

    with torch.no_grad():
        for real_img, gen_img in zip(real_batch, generated_batch):
            real_img = denormalize(real_img).unsqueeze(0).to(device)
            gen_img = denormalize(gen_img).unsqueeze(0).to(device)

            psnr_scores.append(psnr(gen_img, real_img).item())
            ssim_scores.append(ssim(gen_img, real_img).item())
            lpips_scores.append(loss_fn(gen_img, real_img).item())

    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    avg_lpips = sum(lpips_scores) / len(lpips_scores)

    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips
    }