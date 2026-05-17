import os
import torch
import glob

def load_weights(generator, discriminator, weights_dir='./outputs/training/checkpoints'):
    # Assicurati che la cartella esista
    os.makedirs(weights_dir, exist_ok=True)

    # Cerca tutti i file che corrispondono al pattern
    weight_files = glob.glob(os.path.join(weights_dir, 'weights_*.pt'))

    if not weight_files:
        print("Nessun file di pesi trovato. Inizializzazione dei pesi da zero.")
        return 0, {}, {}, {}  # Nessuno step, nessuna loss, nessuna acc, nessuna metrica

    # Ordina i file in base al timestamp nel nome (decrescente, più recente per primo)
    weight_files.sort(reverse=True)
    latest_weight_file = weight_files[0]

    # Carica il checkpoint (usiamo cpu per sicurezza, poi il trainer lo sposterà)
    checkpoint = torch.load(latest_weight_file, map_location='cpu')
    step = checkpoint.get('step', 0)

    # Carica i pesi nei modelli
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    print(f"Pesi caricati con successo da {latest_weight_file}, riprendendo dallo step {step}")

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

    metrics_history = checkpoint.get('metrics_history', [])

    return step + 1, losses, accuracies, metrics_history