import numpy as np

def find_well_trained_step(gen_losses, disc_acc_real, disc_acc_fake, window=1000, gen_loss_thresh=0.01, acc_range=(45, 55)):
    """
    Trova lo step a partire dal quale la rete è considerata sufficientemente addestrata.
    - window: numero di step consecutivi da controllare
    - gen_loss_thresh: variazione massima media del generatore (percentuale)
    - acc_range: intervallo accettabile di accuratezza (%) per il discriminatore
    """

    for i in range(len(gen_losses) - window):
        gen_window = gen_losses[i:i + window]
        acc_real_window = disc_acc_real[i:i + window]
        acc_fake_window = disc_acc_fake[i:i + window]

        gen_var = np.abs((gen_window[-1] - gen_window[0]) / (gen_window[0] + 1e-8))
        acc_real_ok = np.all((np.array(acc_real_window) >= acc_range[0]) & (np.array(acc_real_window) <= acc_range[1]))
        acc_fake_ok = np.all((np.array(acc_fake_window) >= acc_range[0]) & (np.array(acc_fake_window) <= acc_range[1]))

        if gen_var < gen_loss_thresh and acc_real_ok and acc_fake_ok:
            return i + window  # restituisce lo step dove inizia la stabilità
    return None  # non trovato
