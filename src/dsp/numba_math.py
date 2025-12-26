import numpy as np
from numba import jit
from numba.typed import List

@jit(nopython=True)
def blackman_harris_window(size):
    """
    Generates a Blackman-Harris window of the given size.
    """
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168

    window = np.zeros(size, dtype=np.float32)
    for n in range(size):
        term1 = a1 * np.cos(2 * np.pi * n / (size - 1))
        term2 = a2 * np.cos(4 * np.pi * n / (size - 1))
        term3 = a3 * np.cos(6 * np.pi * n / (size - 1))
        window[n] = a0 - term1 + term2 - term3

    return window

@jit(nopython=True)
def apply_window_and_pad(buffer, window, padded_size):
    """
    Applies window and pads buffer with zeros.
    Returns a float32 array ready for FFT.
    """
    out = np.zeros(padded_size, dtype=np.float32)
    # Direct copy with window
    for i in range(len(buffer)):
        out[i] = buffer[i] * window[i]
    return out

@jit(nopython=True)
def spectral_ops_and_detect(fft_magnitude, sample_rate, padded_size, min_threshold=0.05):
    """
    Performs Whitening and Iterative Subtraction.
    Returns list of detected frequencies.
    """
    # 1. Spectral Whitening (Normalize)
    max_val = 0.0
    for i in range(len(fft_magnitude)):
        if fft_magnitude[i] > max_val:
            max_val = fft_magnitude[i]

    if max_val > 0:
        for i in range(len(fft_magnitude)):
            fft_magnitude[i] /= max_val

    # 2. Iterative Subtraction
    detected_frequencies = List()
    freq_res = sample_rate / padded_size

    # Harmonics usually don't go past 6 for guitar processing relevance
    max_notes = 6

    # Copy spectrum to work on it
    work_spec = fft_magnitude.copy()

    for _ in range(max_notes):
        # Find peak
        peak_mag = -1.0
        peak_idx = -1

        # Start searching from ~70Hz (approx bin 3) to avoid DC offset/rumble
        start_bin = 3

        for i in range(start_bin, len(work_spec)):
            if work_spec[i] > peak_mag:
                peak_mag = work_spec[i]
                peak_idx = i

        # Threshold check
        if peak_mag < min_threshold:
            break

        # Add to results
        detected_frequencies.append(peak_idx * freq_res)

        # Kill the fundamental and its harmonics
        fundamental = peak_idx

        # Suppress fundamental (kill zone: +/- 2 bins)
        low = max(0, fundamental - 2)
        high = min(len(work_spec), fundamental + 3)
        work_spec[low:high] = 0.0

        # Suppress harmonics (integer multiples)
        # We assume harmonics up to 5th order
        for h in range(2, 6):
            harmonic_idx = fundamental * h
            if harmonic_idx < len(work_spec):
                # Wider kill zone for harmonics (strings stretch!)
                # Kill +/- 3 bins around harmonic
                h_low = max(0, harmonic_idx - 3)
                h_high = min(len(work_spec), harmonic_idx + 4)
                work_spec[h_low:h_high] = 0.0

    return detected_frequencies

@jit(nopython=True)
def calculate_rms(buffer):
    """
    Calculates the Root Mean Square (RMS) of a buffer.
    """
    sum_squares = 0.0
    for sample in buffer:
        sum_squares += sample * sample
    return np.sqrt(sum_squares / len(buffer))

@jit(nopython=True)
def detect_transient(current_rms, previous_rms, threshold_ratio=2.0, min_rms=0.01):
    """
    Detects a transient if the RMS spike is above a threshold.
    """
    if previous_rms < min_rms:
        # Avoid division by zero or huge spikes from silence
        return current_rms > min_rms * 2

    ratio = current_rms / previous_rms
    return ratio > threshold_ratio
