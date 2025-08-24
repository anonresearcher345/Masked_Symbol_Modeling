import numpy as np
import numpy.typing as npt
from data.data_utils import zero_small_values

def ask_mod(dec_in: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Modulates a given integer-valued array using M-ary
    Amplitude Shift Keying (M-ASK).

    This function implements only unipolar M-ASK.
    Bipolar M-ASK is not included as bipolar 2-ASK
    corresponds to BPSK.

    Args:
        dec_in: A 1D numpy array of integers, e.g., 0, 1, ..., M-1.

    Returns:
        A (2, N) numpy array of float32 type, where the first row
        is the in-phase (I) channel and the second row is the
        quadrature (Q) component (all zeros for ASK).
    """
    dec_in = (2*dec_in)+1
    sym_I = dec_in.astype(np.float32)
    sym_Q = np.zeros_like(sym_I, dtype=np.float32)

    return np.vstack((sym_I, sym_Q))

def psk_mod(dec_in: npt.NDArray[np.uint8], M: int, phase_off: float=0) -> npt.NDArray[np.float32]:
    """Modulates a given integer-valued array using M-ary
    Phase Shift Keying (PSK).

    Args:
        dec_in: A 1D numpy array of integers, e.g., 0, 1, ..., M-1.
        M: The modulation order, e.g., 2 for BPSK, 4 for QPSK, etc.
        phase_off: Optional phase offset in radians to
        rotate the constellation.
    Returns:
        A (2, N) numpy array of float32 type.
        First row - I channel.
        Second row - Q channel.
    """
    sym_I = np.cos((dec_in/M)*2*np.pi + phase_off, dtype=np.float32)
    sym_Q = np.sin((dec_in/M)*2*np.pi + phase_off, dtype=np.float32)
    
    zero_small_values(sym_I)
    zero_small_values(sym_Q)

    return np.vstack((sym_I, sym_Q))

def qam_mod(dec_in: npt.NDArray[np.uint8], M: int) -> npt.NDArray[np.float32]:
    """Modulates a given integer-valued array using square 
    Quadrature Amplitude Modulation (QAM). 

    Args:
        dec_in: A 1D numpy array of integers, e.g., 0, 1, ..., M-1.
        M: The modulation order. Must be a power of 4.
        
    Returns:
        A (2, N) numpy array of float32 type.
        First row - I channel.
        Second row - Q channel.

    Raises:
        ValueError: If M is not a valid power of 4.
    """
    # Check if M is power of 4.
    if (M <= 1) or ((M & (M-1))!=0) or ((M & 0x5555)==0):
        raise ValueError('M must be a power of four.')

    D = int(np.sqrt(M))
    d = np.arange(D)+1
    A = 2*d-1-D
    I_map = np.repeat(-A, repeats=D)
    Q_map = np.tile(A, reps=D)
    sym_I = I_map[dec_in]
    sym_Q = Q_map[dec_in]

    return np.vstack((sym_I, sym_Q), dtype=np.float32)

def modulate(dec_in: npt.NDArray[np.uint8], mod_family: str=None, M: int=None, 
             phase_off: float=0) -> npt.NDArray[np.float32]: 
    """Modulates a given integer-valued array 
    using the specified digital modulation scheme.

    Args:
        dec_in: A 1D numpy array of integers, e.g., 0, 1, ..., M-1.
        mod_family: The modulation family to use 
        ('ask', 'psk', or 'qam').
        M: The modulation order, e.g., 2 for BPSK, 
        4 for QPSK or 4-QAM, etc.).
        phase_off: Optional phase offset in radians, 
        used only for PSK.

    Returns:
        A (2, N) numpy array of float32 type.
        First row - I channel.
        Second row - Q channel.

    Raises:
        TypeError: If 'mod_family' or 'M' is not provided.
        ValueError: If 'mod_family' is not one of the supported types.
    """
    if mod_family is None:
        raise TypeError('Provide a modulation family.')
    if M is None:
        raise TypeError('Provide the modulation order.')
    if mod_family == "ask":
        symbols = ask_mod(dec_in)
    elif mod_family == "psk":
        symbols = psk_mod(dec_in, M, phase_off)
    elif mod_family == "qam":
        symbols = qam_mod(dec_in, M)
    else:
        raise ValueError("Unsupported modulation type.")

    return symbols