import unittest
import numpy as np
from src.dsp.numba_math import calculate_rms, detect_transient

class TestTransientDetector(unittest.TestCase):
    def test_calculate_rms(self):
        # Test 1: Silence
        buffer = np.zeros(128, dtype=np.float32)
        rms = calculate_rms(buffer)
        self.assertEqual(rms, 0.0)

        # Test 2: DC offset 1.0
        buffer = np.ones(128, dtype=np.float32)
        rms = calculate_rms(buffer)
        self.assertAlmostEqual(rms, 1.0)

        # Test 3: Sine wave amplitude 1.0 -> RMS should be 1/sqrt(2) approx 0.707
        t = np.linspace(0, 1, 128)
        buffer = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        rms = calculate_rms(buffer)
        # It won't be exactly 0.707106 due to discretization but close
        self.assertTrue(0.70 < rms < 0.72)

    def test_detect_transient(self):
        # Case 1: Steady state (no transient)
        self.assertFalse(detect_transient(0.5, 0.5))

        # Case 2: Spike (transient)
        self.assertTrue(detect_transient(0.5, 0.1, threshold_ratio=2.0))

        # Case 3: Small changes
        self.assertFalse(detect_transient(0.15, 0.1, threshold_ratio=2.0))

        # Case 4: Rise from silence
        self.assertTrue(detect_transient(0.1, 0.0, min_rms=0.01))

        # Case 5: Noise floor variations
        self.assertFalse(detect_transient(0.005, 0.004, min_rms=0.01))

if __name__ == '__main__':
    unittest.main()
