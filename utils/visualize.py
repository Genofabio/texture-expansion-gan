import torch
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as TF

def denormalize(tensor):
    return (tensor + 1) / 2


def ensure_dir_exists(file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def moving_average(data, window_size):
    if len(data) < window_size:
        return []
    return [sum(data[i:i+window_size])/window_size for i in range(len(data) - window_size + 1)]


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


def visualize_losses(losses_D, losses_G, timestamp, best_step=None):
    plt.figure(figsize=(10,5))
    plt.plot(losses_D, label='Loss Discriminatore')
    plt.plot(losses_G, label='Loss Generatore')
    if len(losses_D) >= 200:
        avg_D = moving_average(losses_D, 200)
        avg_G = moving_average(losses_G, 200)
        plt.plot(range(199, len(losses_D)), avg_D, label='Media 200 Loss D', linestyle='--')
        plt.plot(range(199, len(losses_G)), avg_G, label='Media 200 Loss G', linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Andamento delle loss (start: {timestamp})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_dir='results/losses'
    ensure_dir_exists(save_dir)
    save_path = os.path.join(save_dir, f'loss_plot_{timestamp}.png')
    plt.savefig(save_path)
    plt.close()


def visualize_generator_loss_components(losses_G_adv, losses_L1, losses_style, losses_perceptual, timestamp):
    plt.figure(figsize=(10, 6))
    plt.plot(losses_G_adv, label='Adversarial Loss', color='blue')
    plt.plot(losses_L1, label='L1 Loss', color='green')
    plt.plot(losses_style, label='Style Loss', color='red')
    plt.plot(losses_perceptual, label='Perceptual Loss', color='purple')
    if len(losses_G_adv) >= 200:
        plt.plot(range(199, len(losses_G_adv)), moving_average(losses_G_adv, 200), linestyle='--', label='Media 200 Adv', color='blue')
        plt.plot(range(199, len(losses_L1)), moving_average(losses_L1, 200), linestyle='--', label='Media 200 L1', color='green')
        plt.plot(range(199, len(losses_style)), moving_average(losses_style, 200), linestyle='--', label='Media 200 Style', color='red')
        plt.plot(range(199, len(losses_perceptual)), moving_average(losses_perceptual, 200), linestyle='--', label='Media 200 Percep', color='purple')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Componenti della Loss del Generatore (start: {timestamp})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_dir='results/losses'
    ensure_dir_exists(save_dir)
    save_path = os.path.join(save_dir, f'generator_componnents_plot_{timestamp}.png')
    plt.savefig(save_path)
    plt.close()


def visualize_discriminator_accuracy(acc_real_list, acc_fake_list, timestamp):
    plt.figure(figsize=(10, 5))
    plt.plot(acc_real_list, label='Accuratezza su reali', color='green')
    plt.plot(acc_fake_list, label='Accuratezza su fake', color='red')
    if len(acc_real_list) >= 200:
        plt.plot(range(199, len(acc_real_list)), moving_average(acc_real_list, 200), linestyle='--', label='Media 200 Real', color='green')
        plt.plot(range(199, len(acc_fake_list)), moving_average(acc_fake_list, 200), linestyle='--', label='Media 200 Fake', color='red')
    plt.xlabel('Step')
    plt.ylabel('Accuratezza (%)')
    plt.title(f'Andamento accuratezza Discriminatore (start: {timestamp})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_dir='results/losses'
    ensure_dir_exists(save_dir)
    save_path = os.path.join(save_dir, f"discriminator_accuracy_plot_{timestamp}.png")
    plt.savefig(save_path)
    plt.close()

