# Project: PyPolyGuitar
# File: run.py
import time
from src.config import RING_BUFFER_SIZE
from src.audio.ring_buffer import RingBuffer
from src.audio.input_stream import AudioStream

def main():
    print("Initializing PyPolyGuitar...")

    # 1. Initialize Ring Buffer
    ring_buffer = RingBuffer(RING_BUFFER_SIZE)

    # 2. Initialize Audio Stream
    # Note: Renamed InputStream to AudioStream in input_stream.py
    audio_stream = AudioStream(ring_buffer)

    print("Starting Audio Stream...")
    try:
        audio_stream.start()
        print("Audio Stream Started. Press Ctrl+C to stop.")

        while True:
            # Just keep the main thread alive
            time.sleep(1)
            # Optional: verify buffer is being written to (debug)
            # recent = ring_buffer.read_recent(10)
            # print(f"Buffer snapshot: {recent[:5]}...")

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        audio_stream.stop()
        print("Exited.")

if __name__ == "__main__":
    main()
