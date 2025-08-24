import numpy as np
import numpy.typing as npt

def zero_small_values(arr: np.ndarray, tol: float=1e-6) -> None:
    arr[np.abs(arr) < tol] = 0

# --------------------
# Waveform Power Utilities
# --------------------

def calc_pow(waveform: npt.NDArray[np.float32]) -> float:
    sig_pow = (np.sum((waveform[0, :]**2 + waveform[1, :]**2))
               /np.shape(waveform)[1])
    return sig_pow

def unit_sig_pow(waveform: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    sig_pow = calc_pow(waveform)
    return waveform/np.sqrt(sig_pow)