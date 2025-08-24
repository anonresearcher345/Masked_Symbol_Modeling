import numpy as np
import numpy.typing as npt

from data.data_utils import calc_pow

def add_awgn(waveform: npt.NDArray[np.float32], snr: float=1e-6) -> npt.NDArray[np.float32]:
    """Add circularly-symmetric complex normal noise
    to the given baseband waveform.

    1. Convert snr from dB to linear scale.
    2. Compute signal power in the given waveform.
    3. Compute total variance. 
    (per-branch variance = total variance/2)
    4. Sample independent Gaussian noise
    for each branch (I and Q).
    5. Post-hoc scale with a single factor `k`,
    so that snr_lin = P_sig/P_noise exactly.

    Args:
        waveform: Shape (2, N) waveform samples.
        snr: Target average SNR in dB.

    Returns:
        noisy_waveform: Shape (2, N) waveform
        corrupted by proper circular-symmetric AWGN.
    """
    snr_lin = np.power(10, snr/10)
    waveform_pow = calc_pow(waveform)
    var = waveform_pow /snr_lin
    
    rng1 = np.random.default_rng()
    rng2 = np.random.default_rng()

    N = np.shape(waveform)[1]
    
    noise1 = rng1.normal(loc=0, scale=np.sqrt(var/2), size=N)
    noise2 = rng2.normal(loc=0, scale=np.sqrt(var/2), size=N)
    noise = np.vstack((noise1, noise2), dtype=np.float32)
    k = np.sqrt(waveform_pow / (snr_lin*calc_pow(noise)))
    noise *= k

    noisy_waveform = (waveform + noise).astype(np.float32)
    return noisy_waveform

def add_middleton_a(waveform: npt.NDArray[np.float32],
                    snr: float=1e-6,
                    A: float=1e-2,
                    gamma: float=1e-6) -> npt.NDArray[np.float32]:
    """Add circular Middleton Class A noise.

    One Poisson draw is shared by I and Q. This
    makes the two branches statistically dependent
    but preserves circular symmetry, which is the
    standard modelling choice in the literature.

    1. Split the total noise-power budget
    into the Gaussian and impulsive components.
    2. Sample the shared Poisson counts.
    3. Generate the Gaussian noise with the
    corresponding variance for each branch.
    4. Apply the scalar factor `k` so that the realized
    empirical SNR is exactly equal to the target SNR
    without altering the noise statistics.

    Args:
        waveform: Shape (2, N) waveform samples.
        snr: Total target average SNR in dB.
        A: Impulsive index.
        gamma: Gaussian-to-impulse power ratio.

    Returns:
        noisy_waveform: Shape (2, N) waveform
        corrupted by proper circular-symmetric
        Middleton Class A noise.

    """
    snr_lin = np.power(10, snr/10)
    waveform_pow = calc_pow(waveform)
    total_noise_pow = waveform_pow/snr_lin

    var_impulse = total_noise_pow / (1 + gamma)
    var_norm = var_impulse * gamma

    # Generate the noise samples
    # according to the physical recipe
    # Poisson count of impulses -> Gaussian sample draw
    rng_pois = np.random.default_rng()
    rng_norm1 = np.random.default_rng()
    rng_norm2 = np.random.default_rng()

    N = np.shape(waveform)[1]

    m = rng_pois.poisson(lam=A, size=N)
    var_instant = var_norm * (m/(A*gamma) + 1)

    noise1 = rng_norm1.normal(loc=0, scale=np.sqrt(var_instant/2), size=N)
    noise2 = rng_norm2.normal(loc=0, scale=np.sqrt(var_instant/2), size=N)
    noise = np.vstack((noise1, noise2), dtype=np.float32)
    k = np.sqrt(waveform_pow / (snr_lin*calc_pow(noise)))
    noise *= k
    
    noisy_waveform = (waveform + noise).astype(np.float32)
    return noisy_waveform