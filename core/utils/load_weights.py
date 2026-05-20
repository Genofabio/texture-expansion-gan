import os
import torch
import glob

def load_weights(generator, discriminator, weights_dir='./outputs/training/checkpoints'):
    # Ensure the directory exists
    os.makedirs(weights_dir, exist_ok=True)

    # Search for all files matching the pattern
    weight_files = glob.glob(os.path.join(weights_dir, 'weights_*.pt'))

    if not weight_files:
        print("No weight files found. Initializing weights from scratch.")
        return 0, {}, {}, {}  # No step, no losses, no accuracies, no metrics

    # Sort files based on the timestamp in the name (descending, newest first)
    weight_files.sort(reverse=True)
    latest_weight_file = weight_files[0]

    # Load the checkpoint (use CPU for safety, the trainer will move it later)
    checkpoint = torch.load(latest_weight_file, map_location='cpu')
    step = checkpoint.get('step', 0)

    # Load weights into the models
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    print(f"Weights successfully loaded from {latest_weight_file}, resuming from step {step}")

    # Retrieve metrics history if present
    losses = {
        'losses_D_list': checkpoint.get('losses_D_list', []),
        'losses_G_list': checkpoint.get('losses_G_list', []),
        'losses_G_adv_list': checkpoint.get('losses_G_adv_list', []),
        'losses_L1_list': checkpoint.get('losses_L1_list', []),
        'losses_style_list': checkpoint.get('losses_style_list', []),
        'losses_perceptual_list': checkpoint.get('losses_perceptual_list', [])
    }

    accuracies = {
        'acc_real_list': checkpoint.get('acc_real_list', []),
        'acc_fake_list': checkpoint.get('acc_fake_list', [])
    }

    metrics_history = checkpoint.get('metrics_history', [])

    return step + 1, losses, accuracies, metrics_history