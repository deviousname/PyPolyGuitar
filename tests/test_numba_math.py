import unittest
import numpy as np
from src.dsp.numba_math import (
    blackman_harris_window,
    apply_window_and_pad,
    spectral_ops_and_detect,
)

class TestNumbaMath(unittest.TestCase):
    def test_blackman_harris_window(self):
        size = 128
        window = blackman_harris_window(size)
        self.assertEqual(len(window), size)
        self.assertTrue(np.all(window >= 0))
        self.assertTrue(np.all(window <= 1))

    def test_pipeline_sine_wave(self):
        sample_rate = 48000
        duration = 1.0  # seconds
        freq = 440.0    # A4
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Create a pure sine wave
        sine_wave = np.sin(2 * np.pi * freq * t)

        # Take a chunk
        chunk_size = 512
        buffer = sine_wave[:chunk_size].astype(np.float32)

        # Window
        window = blackman_harris_window(chunk_size)

        # Pad and FFT
        padded_size = 2048

        ready_for_fft = apply_window_and_pad(buffer, window, padded_size)

        # FFT (Standard Numpy)
        fft_complex = np.fft.rfft(ready_for_fft)
        fft_magnitude = np.abs(fft_complex).astype(np.float32)

        # Detect
        detected_freqs = spectral_ops_and_detect(fft_magnitude, sample_rate, padded_size)

        # Check if it's close to 440
        # Resolution is 48000 / 2048 = ~23.4 Hz.
        # So we expect it to be within one bin width.
        found = False
        for f in detected_freqs:
            if abs(f - freq) < (sample_rate / padded_size):
                found = True
                break

        self.assertTrue(found, f"440Hz not found in {detected_freqs}")
        print(f"Detected frequency: {detected_freqs} (Expected ~{freq} Hz)")

    def test_spectral_ops_chord(self):
        sample_rate = 48000
        padded_size = 2048
        freq_res = sample_rate / padded_size

        # Construct a synthetic spectrum
        # Spectrum size for Real FFT is padded_size // 2 + 1
        spectrum_len = padded_size // 2 + 1
        spectrum = np.zeros(spectrum_len, dtype=np.float32)

        # Add fundamental E2 (82.4 Hz) and harmonics
        e2_freq = 82.4
        e2_bin = int(round(e2_freq / freq_res))
        spectrum[e2_bin] = 1.0 # Fundamental
        spectrum[e2_bin * 2] = 0.5 # 2nd Harmonic (164.8)
        spectrum[e2_bin * 3] = 0.3 # 3rd Harmonic

        # Add fundamental A2 (110 Hz) and harmonics
        a2_freq = 110.0
        a2_bin = int(round(a2_freq / freq_res))
        spectrum[a2_bin] = 0.8 # Fundamental
        spectrum[a2_bin * 2] = 0.4 # 2nd Harmonic

        # Normalize is done inside spectral_ops_and_detect, so we don't strictly need to pre-normalize,
        # but the function expects raw magnitude input.

        # Run subtraction logic
        detected_freqs = spectral_ops_and_detect(spectrum, sample_rate, padded_size, min_threshold=0.1)

        print(f"Detected chord frequencies: {detected_freqs}")

        # We expect to find E2 and A2 approx
        # Convert detected freqs back to bins to verify or check range

        # Check if E2 is detected
        e2_detected = any(abs(f - e2_freq) < freq_res for f in detected_freqs)
        self.assertTrue(e2_detected, f"E2 ({e2_freq}Hz) not detected in {detected_freqs}")

        # Check if A2 is detected
        a2_detected = any(abs(f - a2_freq) < freq_res for f in detected_freqs)
        self.assertTrue(a2_detected, f"A2 ({a2_freq}Hz) not detected in {detected_freqs}")

        # Check that harmonics were NOT detected as separate notes
        # e.g. 164.8 Hz should be filtered out
        e2_harmonic_freq = e2_freq * 2
        harmonic_detected = any(abs(f - e2_harmonic_freq) < freq_res for f in detected_freqs)
        self.assertFalse(harmonic_detected, f"Harmonic ({e2_harmonic_freq}Hz) incorrectly detected as note")

if __name__ == '__main__':
    unittest.main()
