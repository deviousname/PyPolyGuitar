import time
import numpy as np
from src.config import RING_BUFFER_SIZE, SAMPLE_RATE
from src.audio.ring_buffer import RingBuffer
from src.audio.input_stream import AudioStream
from src.dsp.numba_math import blackman_harris_window, apply_window_and_pad, spectral_ops_and_detect
from src.midi.interface import MidiInterface

def main():
    # Setup
    ANALYSIS_WINDOW = 512
    PADDED_SIZE = 2048

    ring_buffer = RingBuffer(RING_BUFFER_SIZE)
    audio_stream = AudioStream(ring_buffer)
    midi = MidiInterface()
    midi.open_port()

    # Pre-calc window
    window = blackman_harris_window(ANALYSIS_WINDOW)

    print("Starting Engine...")
    audio_stream.start()

    try:
        while True:
            # 1. Poll Buffer faster (No sleep)
            # Only process if we have enough new data?
            # For simplicity in this version, we just poll.
            # Ideally, we sync this to the callback, but polling fast is OK for V1.

            # Read latest audio
            raw_audio = ring_buffer.read_recent(ANALYSIS_WINDOW)

            # Check RMS (Noise Gate)
            rms = np.sqrt(np.mean(raw_audio**2))
            if rms < 0.002: # Silence threshold
                midi.update_notes([]) # Clear notes if silent
                time.sleep(0.001) # Tiny sleep to save CPU on silence
                continue

            # 2. Pre-process (Numba)
            ready_for_fft = apply_window_and_pad(raw_audio, window, PADDED_SIZE)

            # 3. FFT (Standard Numpy - Fast C implementation)
            fft_complex = np.fft.rfft(ready_for_fft)
            fft_magnitude = np.abs(fft_complex).astype(np.float32)

            # 4. Detect (Numba)
            detected_freqs = spectral_ops_and_detect(fft_magnitude, SAMPLE_RATE, PADDED_SIZE)

            # 5. MIDI
            midi.update_notes(detected_freqs)

            # Tiny sleep to prevent 100% CPU usage, but low enough for latency
            # 1ms sleep = 1000Hz update rate
            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        audio_stream.stop()
        midi.close_port()

if __name__ == "__main__":
    main()
