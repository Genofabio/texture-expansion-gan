import os
import torch

def load_weights(generator, discriminator, weights_path='data/weights_to_use.pt'):
    if not os.path.exists(weights_path):
        print("File weights_to_use.pt non trovato. Inizializzazione dei pesi.")
        return 0, {}, {}  # Nessuno step, nessuna loss, nessuna acc

    checkpoint = torch.load(weights_path, map_location='cpu')
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    print(f"Pesi caricati da {weights_path}, step {checkpoint.get('step', 0)}")

    # Recupera metriche se presenti
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

    return checkpoint.get('step', 0) + 1, losses, accuracies