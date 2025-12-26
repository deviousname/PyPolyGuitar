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
def process_fft(buffer, window, padded_size):
    """
    Applies window, zero-pads, and computes Real FFT.
    """
    # 1. Apply Window
    n = len(buffer)
    # We assume window size matches buffer size or is handled by caller
    windowed_buffer = np.zeros(n, dtype=np.float32)
    for i in range(n):
        windowed_buffer[i] = buffer[i] * window[i]

    # 2. Zero-Padding
    # Create a zero-filled array of padded_size
    padded_buffer = np.zeros(padded_size, dtype=np.float32)
    # Copy the windowed buffer into the start
    padded_buffer[:n] = windowed_buffer

    # 3. FFT
    # numpy.fft.rfft returns complex numbers
    # Since numpy.fft is not supported in nopython mode, we use objmode
    # rfft output size is N/2 + 1

    # Define the output variable to hold the result
    # It seems we can just declare the expected type in the context manager

    # We need to calculate the expected output size for type signature matching if needed,
    # but objmode mainly cares about the variable being assigned to.

    # NOTE: float32 input to rfft produces complex64 output in numpy
    with objmode(fft_result='complex64[:]'):
        fft_result = np.fft.rfft(padded_buffer)
        # Ensure it is complex64
        fft_result = fft_result.astype(np.complex64)

    return fft_result

@jit(nopython=True)
def magnitude_spectrum(fft_complex):
    """
    Computes the magnitude of the complex FFT result.
    """
    n = len(fft_complex)
    magnitude = np.zeros(n, dtype=np.float32)
    for i in range(n):
        magnitude[i] = np.abs(fft_complex[i])
    return magnitude

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
def iterative_spectral_subtraction(spectrum, sample_rate, padded_size, min_threshold=0.1, max_notes=6):
    """
    Detects multiple notes by finding the loudest peak and subtracting its harmonics.
    """
    # Create a copy of the spectrum to modify
    working_spectrum = spectrum.copy()

    detected_frequencies = List()

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
