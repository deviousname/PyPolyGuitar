# Project: PyPolyGuitar
# File: numba_math.py
# Description: Numba-accelerated FFT and DSP functions.

import numpy as np
from numba import jit, objmode
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
def process_fft(buffer, window, padded_size, padded_buffer_out):
    """
    Applies window, zero-pads, and computes Real FFT using pre-allocated buffer.
    padded_buffer_out: float32 array of size padded_size
    """
    # 1. Apply Window
    n = len(buffer)
    # Reuse padded_buffer_out for windowed part as well, or just write directly

    # Zero out the padded buffer first (or at least the part we don't write to, if we care)
    # But since we overwrite the beginning and the rest should be zero,
    # and we reuse it... we must ensure the tail is zero.
    # It's better to clear it if we reuse it.
    padded_buffer_out[:] = 0.0

    # Apply window and write to padded_buffer_out
    for i in range(n):
        padded_buffer_out[i] = buffer[i] * window[i]

    # 3. FFT
    # numpy.fft.rfft returns complex numbers
    # Since numpy.fft is not supported in nopython mode, we use objmode

    # NOTE: float32 input to rfft produces complex64 output in numpy
    with objmode(fft_result='complex64[:]'):
        fft_result = np.fft.rfft(padded_buffer_out)
        # Ensure it is complex64
        fft_result = fft_result.astype(np.complex64)

    return fft_result

@jit(nopython=True)
def magnitude_spectrum(fft_complex, magnitude_out):
    """
    Computes the magnitude of the complex FFT result into magnitude_out.
    """
    n = len(fft_complex)
    # Ensure magnitude_out is large enough
    for i in range(n):
        magnitude_out[i] = np.abs(fft_complex[i])
    return magnitude_out

@jit(nopython=True)
def spectral_whitening(spectrum):
    """
    Normalizes the spectrum.
    """
    max_val = 0.0
    for i in range(len(spectrum)):
        if spectrum[i] > max_val:
            max_val = spectrum[i]

    if max_val > 1e-9:
        for i in range(len(spectrum)):
            spectrum[i] = spectrum[i] / max_val

    return spectrum

@jit(nopython=True)
def get_dominant_frequency(spectrum, sample_rate, padded_size):
    """
    Finds the frequency with the highest magnitude.
    """
    max_mag = -1.0
    max_index = -1

    # Skip DC component (index 0) and maybe very low freqs (index 1)
    # let's start from 1 to avoid DC
    for i in range(1, len(spectrum)):
        if spectrum[i] > max_mag:
            max_mag = spectrum[i]
            max_index = i

    if max_index == -1:
        return 0.0

    # Calculate frequency
    # Frequency resolution = sample_rate / padded_size
    # freq = index * resolution
    freq = max_index * (sample_rate / padded_size)

    return freq

@jit(nopython=True)
def iterative_spectral_subtraction(spectrum, sample_rate, padded_size, detected_frequencies_out, min_threshold=0.1, max_notes=6):
    """
    Detects multiple notes by finding the loudest peak and subtracting its harmonics.
    Stores results in detected_frequencies_out (should be a typed List or cleared list).
    Note: 'spectrum' is modified in place (it serves as working_spectrum).
    """
    # We assume the caller passed a copy if they wanted to preserve the original.
    # But here we treat 'spectrum' as mutable working buffer.

    # Clear output list
    # Numba typed list doesn't have clear(), so we re-assign or pop?
    # actually detected_frequencies_out should be passed in empty or we empty it.
    # Numba List: .clear() exists in recent versions? Or we just assume it's new.
    # The caller should pass a fresh list or we return a new one?
    # "detected_frequencies = List()" allocates.
    # If we want to reuse, we need to pass it.

    # Reuse passed list
    # Clear existing items
    while len(detected_frequencies_out) > 0:
        detected_frequencies_out.pop()

    detected_frequencies = detected_frequencies_out

    # But wait, we can't easily modify spectrum in place if it's reused for visualization later?
    # run.py doesn't use it for anything else.
    # So we can modify 'spectrum' directly.

    working_spectrum = spectrum

    # Frequency resolution
    freq_res = sample_rate / padded_size

    # Minimum magnitude threshold (relative to normalized spectrum 0-1)

    for _ in range(max_notes):
        # 1. Find loudest peak
        max_mag = -1.0
        max_index = -1

        # Skip DC (0) and extremely low frequencies (e.g. < 40Hz)
        # 40Hz / (48000/2048) = 40 / 23.4 = ~1.7 bins
        start_bin = int(40 / freq_res) + 1

        for i in range(start_bin, len(working_spectrum)):
            if working_spectrum[i] > max_mag:
                max_mag = working_spectrum[i]
                max_index = i

        # 2. Check if peak is above noise floor
        if max_mag < min_threshold or max_index == -1:
            break

        # 3. Register Note
        fundamental_freq = max_index * freq_res
        detected_frequencies.append(fundamental_freq)

        # 4. Subtract Harmonics
        # We need to subtract the fundamental and its harmonics from the working spectrum.
        # Simple subtraction strategy: reduce magnitude of harmonic bins.

        # Harmonic series: f, 2f, 3f, 4f...
        # We assume harmonics go up to Nyquist (or array bounds)

        num_harmonics = 10 # subtract first 10 harmonics

        for h in range(1, num_harmonics + 1):
            harmonic_freq = fundamental_freq * h
            harmonic_bin = int(round(harmonic_freq / freq_res))

            if harmonic_bin < len(working_spectrum):
                # Apply suppression window around the bin (to account for leakage)
                # Simple: 3 bins (center, left, right)

                # Center
                working_spectrum[harmonic_bin] *= 0.1 # Heavily suppress

                # Neighbors
                if harmonic_bin > 0:
                    working_spectrum[harmonic_bin - 1] *= 0.3
                if harmonic_bin < len(working_spectrum) - 1:
                    working_spectrum[harmonic_bin + 1] *= 0.3

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
