import os
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure  
import lpips
from torchvision import transforms
from PIL import Image
from utils.visualize import ensure_dir_exists
import matplotlib.pyplot as plt
from datetime import datetime

#from pytorch_fid import fid_score

def save_intraset_metrics_plot(metrics_history, dataset_dim):
    """
    Salva un grafico riassuntivo delle metriche PSNR, SSIM, LPIPS, FID durante il training.
    metrics_history è una lista di dict, es:
    [{'step': 0, 'psnr': ..., 'ssim': ..., 'lpips': ..., 'fid': ...}, ...]
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    steps = [m['step'] for m in metrics_history]
    psnr = [m['psnr'] for m in metrics_history]
    ssim = [m['ssim'] for m in metrics_history]
    lpips = [m['lpips'] for m in metrics_history]
    fid = [m['fid'] for m in metrics_history]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Intraset Metrics Summary ({timestamp})", fontsize=18)

    # PSNR
    axs[0, 0].plot(steps, psnr, color='blue')
    axs[0, 0].set_title('PSNR')
    axs[0, 0].set_xlabel('Step')
    axs[0, 0].set_ylabel('PSNR (dB)')
    axs[0, 0].grid(True)

    # SSIM
    axs[0, 1].plot(steps, ssim, color='green')
    axs[0, 1].set_title('SSIM')
    axs[0, 1].set_xlabel('Step')
    axs[0, 1].set_ylabel('SSIM')
    axs[0, 1].grid(True)

    # LPIPS
    axs[1, 0].plot(steps, lpips, color='red')
    axs[1, 0].set_title('LPIPS')
    axs[1, 0].set_xlabel('Step')
    axs[1, 0].set_ylabel('LPIPS (lower is better)')
    axs[1, 0].grid(True)

    # FID
    axs[1, 1].plot(steps, fid, color='purple')
    axs[1, 1].set_title('FID')
    axs[1, 1].set_xlabel('Step')
    axs[1, 1].set_ylabel('FID (lower is better)')
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_dir='../results/training/intraset'
    ensure_dir_exists(save_dir)
    save_path = os.path.join(save_dir, f"intraset_metrics_dataset_dim_{dataset_dim}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Salvato Intraset metrics plot in: {save_path}")

def denormalize(tensor):
    return (tensor + 1) / 2  # [-1, 1] -> [0, 1]

def evaluate_intraset(real_batch, generated_batch, device='cuda'):
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    loss_fn = lpips.LPIPS(net='vgg').to(device)

    psnr_scores, ssim_scores, lpips_scores = [], [], []

    with torch.no_grad():  # 🔒 evita calcolo del grafo
        for real_img, gen_img in zip(real_batch, generated_batch):
            # Denormalizza da [-1, 1] a [0, 1]
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
        'lpips': avg_lpips,
        'fid': None  # oppure 'not_computed'
    }

def evaluate_intraset_cached(real_batch, generated_batch, device='cuda'):
    if not hasattr(evaluate_intraset_cached, "initialized"):
        evaluate_intraset_cached.psnr = PeakSignalNoiseRatio().to(device)
        evaluate_intraset_cached.ssim = StructuralSimilarityIndexMeasure().to(device)
        evaluate_intraset_cached.loss_fn = lpips.LPIPS(net='vgg').to(device)
        evaluate_intraset_cached.initialized = True

    psnr = evaluate_intraset_cached.psnr
    ssim = evaluate_intraset_cached.ssim
    loss_fn = evaluate_intraset_cached.loss_fn

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
        'lpips': avg_lpips,
        'fid': None
    }

