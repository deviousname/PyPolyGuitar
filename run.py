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
    iterative_spectral_subtraction,
    calculate_rms,
    detect_transient
)
from src.midi.interface import MidiInterface
import numpy as np
from numba.typed import List

def main():
    print("Initializing PyPolyGuitar...")

    # Configuration
    ANALYSIS_WINDOW_SIZE = 512  # Amount of samples to take for analysis
    PADDED_SIZE = 2048          # FFT Size (Zero-padded)

    # 1. Initialize Ring Buffer
    ring_buffer = RingBuffer(RING_BUFFER_SIZE)

    # 2. Initialize Audio Stream
    audio_stream = AudioStream(ring_buffer)

    # 3. Initialize MIDI
    midi = MidiInterface()
    midi.open_port()

    # Pre-calculate window
    window = blackman_harris_window(ANALYSIS_WINDOW_SIZE)

    # Pre-allocate DSP buffers
    padded_buffer_out = np.zeros(PADDED_SIZE, dtype=np.float32)
    # FFT output is complex64, size N/2 + 1
    fft_out_size = PADDED_SIZE // 2 + 1
    magnitude_out = np.zeros(fft_out_size, dtype=np.float32)

    # Detected frequencies out (Numba List)
    # We pass a dummy list but the function currently returns a new list
    detected_frequencies_out = List()

    # State for transient detection
    previous_rms = 0.0

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

                # Check for transient
                # For transient detection, we might want a smaller window (e.g. 128)
                # But here we are using the snapshot.
                # Let's take the last 128 samples for transient detection logic as per Module D description
                recent_chunk = buffer_snapshot[-128:]
                current_rms = calculate_rms(recent_chunk)

                is_transient = detect_transient(current_rms, previous_rms)
                previous_rms = current_rms

                # 4. Perform DSP
                fft_complex = process_fft(buffer_snapshot, window, PADDED_SIZE, padded_buffer_out)
                magnitude = magnitude_spectrum(fft_complex, magnitude_out)
                whitened_spectrum = spectral_whitening(magnitude)

                detected_freqs = iterative_spectral_subtraction(whitened_spectrum, SAMPLE_RATE, PADDED_SIZE, detected_frequencies_out)

                # 5. Output
                # Format list of floats to string
                freq_str = ", ".join([f"{f:6.1f}" for f in detected_freqs])

                transient_msg = " [TRANSIENT]" if is_transient else ""
                print(f"\rDetected Frequencies: [{freq_str}] Hz {transient_msg}       ", end="", flush=True)

                # 6. MIDI Output
                # We only send MIDI updates if we detected something meaningful?
                # Or we update continuously.
                # If a transient is detected, we might want to force an update or handle it specifically.
                # For now, we just pass the detected frequencies to the MIDI interface.
                midi.update_notes(detected_freqs)

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
        if midi.output_port:
            midi.close_port()
        print("Exited.")

if __name__ == "__main__":
    main()
