import torch
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as TF

def denormalize(tensor):
    # Da [-1, 1] a [0, 1] per la visualizzazione
    return (tensor + 1) / 2

def ensure_dir_exists(file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def visualize_sample(source, target, output, step=0, save_dir='results/sample'):
    source_detached = source.detach().cpu()
    target_detached = target.detach().cpu()
    output_detached = output.detach().cpu()

    source_img = denormalize(source_detached[0]).clamp(0, 1)
    target_img = denormalize(target_detached[0]).clamp(0, 1)
    output_img = denormalize(output_detached[0]).clamp(0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Training Step: {step}", fontsize=16)

    axs[0].imshow(TF.to_pil_image(source_img))
    axs[0].set_title("Input 128x128")

    axs[1].imshow(TF.to_pil_image(target_img))
    axs[1].set_title("Target 256x256")

    axs[2].imshow(TF.to_pil_image(output_img))
    axs[2].set_title("Generated 256x256")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"sample_step{step:05}.png")
    ensure_dir_exists(save_path)
    plt.savefig(save_path)
    plt.close(fig)

def visualize_losses(losses_D, losses_G, save_path='results/losses/loss_plot.png'):
    plt.figure(figsize=(10,5))
    plt.plot(losses_D, label='Loss Discriminatore')
    plt.plot(losses_G, label='Loss Generatore')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Andamento delle loss del generatore e del discriminatore')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ensure_dir_exists(save_path)
    plt.savefig(save_path)
    plt.close()

def visualize_generator_loss_components(
    losses_G_adv, losses_L1, losses_style, losses_perceptual, 
    save_path='results/losses/generator_components_plot.png'
):
    plt.figure(figsize=(10, 6))
    plt.plot(losses_G_adv, label='Adversarial Loss', color='blue')
    plt.plot(losses_L1, label='L1 Loss', color='green')
    plt.plot(losses_style, label='Style Loss', color='red')
    plt.plot(losses_perceptual, label='Perceptual Loss', color='purple')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Componenti della Loss del Generatore')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ensure_dir_exists(save_path)
    plt.savefig(save_path)
    plt.close()


def visualize_discriminator_accuracy(acc_real_list, acc_fake_list, save_path='results/losses/discriminator_accuracy.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(acc_real_list, label='Accuratezza su reali', color='green')
    plt.plot(acc_fake_list, label='Accuratezza su fake', color='red')
    plt.xlabel('Step')
    plt.ylabel('Accuratezza (%)')
    plt.title('Andamento accuratezza del Discriminatore')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ensure_dir_exists(save_path)
    plt.savefig(save_path)
    plt.close()


