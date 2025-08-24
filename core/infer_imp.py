import numpy as np

def add_awgn_vec(waveform: np.ndarray, snr: float) -> np.ndarray:
    # TODO: DOUBLE CHECK AND TEST THE IMPLEMENTATION.

    snr_lin = 10.0 ** (snr / 10.0)

    # per-waveform signal power
    power = np.mean(
        waveform[..., 0, :] ** 2 + waveform[..., 1, :] ** 2,
        axis=-1, keepdims=True
    )

    scale = np.sqrt(power / (2 * snr_lin))[..., None]

    rng = np.random.default_rng()
    noise = rng.normal(0.0, scale, size=waveform.shape).astype(np.float32)

    return (waveform + noise).astype(np.float32)


def add_middleton_a_vec(
    waveform: np.ndarray,
    snr: float,
    A: float = 1e-2,
    gamma: float = 1e-6,
    return_m: bool = False,
) -> np.ndarray:
    """
    Add proper circular Middleton-Class-A noise to (..., 2, N) waveforms.

    Each waveform in the (potentially multi-dimensional) batch receives its
    *own* Poisson draw sequence, power estimate, and scaling – so the realised
    SNR for every waveform matches the target `snr` in dB.

    Args:
        waveform : ndarray, float32, shape (..., 2, N)
            I/Q samples; last two axes are (channel, time).
        snr : float
            Desired average SNR in dB *per waveform*.
        A : float
            Impulsive index (average number of impulses per sample).
        gamma : float
            Gaussian-to-impulse power ratio.

    Returns:
        noisy_waveform : ndarray, same shape as input, float32
    """
    snr_lin = 10.0 ** (snr / 10.0)
    # per-waveform signal power
    sig_pow = np.mean(waveform[..., 0, :]**2 +
                      waveform[..., 1, :]**2, axis=-1, keepdims=True)

    total_noise_pow = sig_pow / snr_lin
    var_imp = total_noise_pow / (1.0 + gamma)
    var_gauss = var_imp * gamma

    rng = np.random.default_rng()

    *batch_shape, _, N = waveform.shape
    
    m = rng.poisson(lam=A, size=(*batch_shape, N))

    var_inst = var_gauss * (m / (A * gamma) + 1.0)

    noise_I = rng.normal(0.0, np.sqrt(var_inst / 2.0)).astype(np.float32)
    noise_Q = rng.normal(0.0, np.sqrt(var_inst / 2.0)).astype(np.float32)

    noise = np.stack((noise_I, noise_Q), axis=-2)

    noise_pow = np.mean(noise[..., 0, :]**2 + noise[..., 1, :]**2,
                        axis=-1, keepdims=True)

    k = np.sqrt(sig_pow / (snr_lin * noise_pow))[..., None]
    noise *= k
    noisy_wav = (waveform + noise).astype(np.float32)

    return (noisy_wav, m) if return_m else noisy_wav