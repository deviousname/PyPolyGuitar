import sys
import time
import numpy as np
from src.config import SAMPLE_RATE, BUFFER_SIZE

# Try importing sounddevice, mock it if PortAudio is missing (e.g. in CI/Sandbox)
try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except OSError:
    print("Warning: PortAudio library not found. Using Mock SoundDevice.", file=sys.stderr)
    _HAS_SOUNDDEVICE = False

    class MockStream:
        def __init__(self, device=None, channels=1, samplerate=48000, blocksize=128, callback=None, dtype='float32'):
            self.callback = callback
            self.blocksize = blocksize
            self.channels = channels
            self.active = False

        def start(self):
            self.active = True
            import threading
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._run)
            self._thread.start()

        def _run(self):
            while self.active and not self._stop_event.is_set():
                # Simulate input data
                indata = np.zeros((self.blocksize, self.channels), dtype=np.float32)
                # Output buffer to be filled
                outdata = np.zeros((self.blocksize, self.channels), dtype=np.float32)

                if self.callback:
                    # Callback signature: indata, outdata, frames, time, status
                    self.callback(indata, outdata, self.blocksize, None, None)

                time.sleep(self.blocksize / 48000.0)

        def stop(self):
            self.active = False
            if hasattr(self, '_stop_event'):
                self._stop_event.set()
            if hasattr(self, '_thread'):
                self._thread.join()

        def close(self):
            pass

    # Mock sd module
    class MockSD:
        Stream = MockStream
        def query_devices(self):
            return [{'name': 'Mock Device', 'max_input_channels': 2, 'max_output_channels': 2}]

    sd = MockSD()

class AudioStream:
    def __init__(self, ring_buffer):
        self.ring_buffer = ring_buffer
        self.stream = None
        self.device_id = self._find_asio_device()

    def _find_asio_device(self):
        """
        Attempts to find a Behringer ASIO device.
        Falls back to default input if not found.
        """
        devices = sd.query_devices()
        asio_device = None

        # Look for Behringer and ASIO
        for i, dev in enumerate(devices):
            if "ASIO" in dev['name'] and "Behringer" in dev['name']:
                asio_device = i
                break

        # Fallback 1: Just ASIO
        if asio_device is None:
            for i, dev in enumerate(devices):
                if "ASIO" in dev['name']:
                    asio_device = i
                    break

        if asio_device is not None:
            print(f"Using ASIO Device: {devices[asio_device]['name']}")
            return asio_device
        else:
            print("ASIO device not found. Using default audio device.")
            return None # Use default

    def callback(self, indata, outdata, frames, time, status):
        """
        Audio callback for full duplex stream.
        """
        if status:
            print(status, file=sys.stderr)

        # Pass through: copy input to output
        outdata[:] = indata

        # indata is (frames, channels). We assume mono (extract channel 0).
        # Ensure it's flattened
        if indata.ndim > 1:
            data = indata[:, 0]
        else:
            data = indata

        self.ring_buffer.write(data)

    def start(self):
        """
        Starts the audio stream.
        """
        # Using Stream for full duplex (Input + Output)
        self.stream = sd.Stream(
            device=self.device_id,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BUFFER_SIZE,
            callback=self.callback,
            dtype='float32'
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
