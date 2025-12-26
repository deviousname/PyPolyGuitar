# Project: PyPolyGuitar
# File: run.py
import time
from src.config import RING_BUFFER_SIZE, SAMPLE_RATE
from src.audio.ring_buffer import RingBuffer
from src.audio.input_stream import AudioStream
from src.dsp.numba_math import (
    blackman_harris_window,
    process_fft,
    magnitude_spectrum,
    spectral_whitening,
    iterative_spectral_subtraction
)

def main():
    print("Initializing PyPolyGuitar...")

    # Configuration
    ANALYSIS_WINDOW_SIZE = 512  # Amount of samples to take for analysis
    PADDED_SIZE = 2048          # FFT Size (Zero-padded)

    # 1. Initialize Ring Buffer
    ring_buffer = RingBuffer(RING_BUFFER_SIZE)

    # 2. Initialize Audio Stream
    audio_stream = AudioStream(ring_buffer)

    # Pre-calculate window
    window = blackman_harris_window(ANALYSIS_WINDOW_SIZE)

    print("Starting Audio Stream...")
    try:
        audio_stream.start()
        print("Audio Stream Started. Press Ctrl+C to stop.")

        while True:
            # 3. Read from Ring Buffer
            # We want the most recent samples
            try:
                # We need at least ANALYSIS_WINDOW_SIZE samples written
                # but read_recent handles whatever is there, might duplicate if not enough?
                # The ring buffer logic wraps around.
                buffer_snapshot = ring_buffer.read_recent(ANALYSIS_WINDOW_SIZE)

                # Check if we have enough data (it returns zero initialized buffer initially)
                # But it always returns requested size.

                # 4. Perform DSP
                fft_complex = process_fft(buffer_snapshot, window, PADDED_SIZE)
                magnitude = magnitude_spectrum(fft_complex)
                whitened_spectrum = spectral_whitening(magnitude)

                detected_freqs = iterative_spectral_subtraction(whitened_spectrum, SAMPLE_RATE, PADDED_SIZE)

                # 5. Output
                # Format list of floats to string
                freq_str = ", ".join([f"{f:6.1f}" for f in detected_freqs])
                print(f"\rDetected Frequencies: [{freq_str}] Hz        ", end="", flush=True)

            except Exception as e:
                print(f"\nError in processing loop: {e}")

            # Sleep to reduce CPU usage and console spam
            # ~10 updates per second
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        audio_stream.stop()
        print("Exited.")

if __name__ == "__main__":
    main()
