import numpy as np

def find_well_trained_step(gen_losses, disc_acc_real, disc_acc_fake, window=1000, gen_loss_thresh=0.01, acc_range=(45, 55)):
    """
    Finds the step from which the network is considered sufficiently trained and stable.
    - window: number of consecutive steps to check
    - gen_loss_thresh: maximum allowed average variation for the generator (percentage)
    - acc_range: acceptable accuracy range (%) for the discriminator
    """

    # BUG FIX: Added +1 to ensure the last window is checked, and to prevent 
    # empty ranges when len(gen_losses) exactly equals the window size.
    for i in range(len(gen_losses) - window + 1):
        gen_window = gen_losses[i:i + window]
        acc_real_window = disc_acc_real[i:i + window]
        acc_fake_window = disc_acc_fake[i:i + window]

        # Calculate relative variation between the start and end of the window
        gen_var = np.abs((gen_window[-1] - gen_window[0]) / (gen_window[0] + 1e-8))
        
        # Verify if all discriminator steps within the window fall inside the acceptable accuracy range
        acc_real_ok = np.all((np.array(acc_real_window) >= acc_range[0]) & (np.array(acc_real_window) <= acc_range[1]))
        acc_fake_ok = np.all((np.array(acc_fake_window) >= acc_range[0]) & (np.array(acc_fake_window) <= acc_range[1]))

        if gen_var < gen_loss_thresh and acc_real_ok and acc_fake_ok:
            return i + window  # Returns the step index where stability begins
            
    return None  # Stability criteria not met within the current history