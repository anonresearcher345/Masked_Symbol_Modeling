import numpy as np
import numpy.typing as npt

def design_rcos_filt(beta: float=0.5, filt_span_in_sym: int=1, 
    sps: int=2, Rs: int=1, gain: float=1.0
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Designs a raised cosine (RC) filter impulse response.

    This implementation includes correct handling of the singularity at
    t = (plus/minus) 1/(2* beta * Rs).

    Note:
        The parameter 'sps' must be even. This guarantees that:
            filt_span_in_samp = filt_span_in_sym * sps -> even
            filt_length = filt_span_in_samp + 1 -> odd
        An odd number of taps ensures the impulse response is symmetric around
        its center, which is required for a linear-phase FIR filter to prevent
        phase/delay distortion.

    Args:
        beta: Roll-off factor (0 <= beta <= 1).
        filt_span_in_sym: Filter span in symbols. The total number of taps
        will be (filt_span_in_sym * sps + 1).
        sps: Samples per symbol (oversampling factor) (must be an even number).
        Rs: Symbol rate in symbols per second.
        gain: Linear gain applied to the filter output.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - t: Time vector centered at 0.
            - h: Impulse response of the RC filter.
    """
    
    beta = np.float64(beta)
    Rs = np.float64(Rs)
    gain = np.float64(gain)

    time_width = filt_span_in_sym/Rs
    filt_span_in_samp = filt_span_in_sym*sps

    # Create symmetric time vector centered at 0
    t = np.linspace(-time_width/2, time_width/2, filt_span_in_samp+1, 
                    dtype=np.float64)
    
    # Special case: beta = 0 -> Ideal sinc filter
    if beta == 0:
        return t.astype(np.float32), gain*(Rs*np.sinc(t*Rs).astype(np.float32))

    h = np.zeros_like(t, dtype=np.float64)
    
    # Identify singularities
    rtol = 1e-5
    atol = 1e-8
    singular_mask = np.isclose(np.abs(t), 1/(2*beta*Rs), 
                               rtol=rtol, atol=atol)
    # Assign finite values at singularities using the limit
    # of the function at the singularity points
    h[singular_mask] = Rs*(np.pi/4)*np.sinc(1/(2*beta))

    # Compute the impulse response for all non-singular time values
    valid_mask = ~singular_mask
    t_valid = t[valid_mask]
    tRs = t_valid*Rs

    h[valid_mask] = (
        Rs*np.sinc(tRs)*
        np.cos(np.pi*beta*tRs)/
        (1 - (2*beta*tRs)**2)
    )

    return t.astype(np.float32), (gain*h).astype(np.float32)

def apply_filt(symbols: npt.NDArray[np.float32], 
               filt_span_in_sym: int, 
               sps: int, 
               h:npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Upsamples and filters I/Q baseband symbols using the given
    impulse response h. The filtering is done separately on the I and Q branches.

    Assumptions:
        - The filter 'h' is symmetric and has odd length.
        - The group delay is (len(h) - 1)//2 samples.
        - The 'sps' parameter used here must match that used to design the filter.
        - The FIR filter h introduces a symmetric delay that is removed to center
        the output waveform.

    Args:
        symbols: Array of shape (2, N), containing I and Q baseband symbols.
        filt_span_in_sym: Filter span in symbols (used to calculate delay).
        sps: Samples per symbol (upsampling factor).
        h: 1D FIR filter (must be odd-length, symmetric).

    Returns:
        np.ndarray: Filtered I/Q waveform of shape.

    """
    # Upsample: insert zeros between each symbol
    upsampled = np.zeros((2, np.shape(symbols)[1]*sps), dtype=np.float32)
    upsampled[0, ::sps] = symbols[0, :]
    upsampled[1, ::sps] = symbols[1, :]
    
    # Apply the filter
    waveform_I = np.convolve(upsampled[0, :], h)
    waveform_Q = np.convolve(upsampled[1, :], h)

    # Remove the leading and trailing transients due to filter delay
    # This aligns the output waveform with the original symbol timing
    num_samples = np.shape(waveform_I)[0]
    filt_delay_in_samp = filt_span_in_sym*sps//2

    waveform_I = waveform_I[filt_delay_in_samp:num_samples-filt_delay_in_samp]
    waveform_Q = waveform_Q[filt_delay_in_samp:num_samples-filt_delay_in_samp]

    return np.vstack((waveform_I, waveform_Q)).astype(np.float32)