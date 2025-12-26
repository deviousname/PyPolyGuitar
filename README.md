# PyPolyGuitar

Real-time Polyphonic Guitar-to-MIDI Converter using Numba-accelerated FFT.

## Setup
1. `pip install -r requirements.txt`
2. `python run.py`

Here is the Architectural Blueprint for the "PyPolyGuitar" Engine. Hand this to your coding team.

Project Name: PyPolyGuitar (Low-Latency ASIO MIDI Converter)
1. Executive Summary

A headless, console-based Python application that utilizes the Behringer UM2 (ASIO) to process raw guitar audio in real-time. It uses Numba-accelerated Fourier transforms to detect multiple simultaneous frequencies (chords), filters out harmonic overtones, and outputs MIDI events to a virtual port (loopMIDI) for use in games like Rust.

2. Core Constraints & Targets

Latency Budget: < 15ms total roundtrip (Buffer size: 128 or 256 samples @ 48kHz).

Input: Mono Audio via ASIO (Behringer UM2, Channel 2).

Output: Standard MIDI Note On/Off messages.

Processing: Must identify fundamental frequencies while suppressing harmonics.

3. Component Breakdown
Module A: The Audio Ingest (ASIO Wrapper)

Python's standard audio libraries are too slow (WASAPI/MME). We require a direct ASIO stream.

Library: sounddevice (Must be installed with ASIO support) or pyaudio (compiled with PortAudio/ASIO).

Requirements:

Callback Mode: Must use a non-blocking callback function. Do not use blocking .read() calls.

Ring Buffer: Implement a numpy ring buffer (circular buffer) to store the last ~2048 samples for analysis, while stepping forward in chunks of 128 samples.

Bit Depth: 32-bit float (normalized -1.0 to 1.0).

Module B: The Signal Processor (Numba/Math)

This is the heavy lifter. All functions here must be decorated with @jit(nopython=True) to bypass the Python Global Interpreter Lock (GIL).

1. Windowing Function:

Apply a Blackman-Harris window to the audio buffer before FFT. This reduces "spectral leakage" (where one note smears into neighbor frequencies).

2. Zero-Padding:

Pad the buffer with zeros to increase FFT resolution without increasing latency (e.g., take 512 samples, pad to 2048).

3. FFT (Fast Fourier Transform):

Use numpy.fft.rfft (Real input FFT) inside a Numba function.

Convert complex numbers to Magnitude Spectrum (Volume per frequency).

4. Spectral Whitening (The "Chord" Secret):

Guitar signals have massive energy in low mids. We must normalize the spectrum so high notes (which are quieter) can be seen next to loud bass notes.

Module C: The "Chord Solver" Algorithm

This is the logic that beats the other software. It filters out the "Ghost Notes" (Harmonics).

Algorithm: Iterative Spectral Subtraction

Find the loudest peak in the spectrum (e.g., 82Hz).

Check if it is a transient (did it just appear?).

Register Note: E2.

The Math Trick: Calculate where the harmonics of E2 live (164Hz, 246Hz, 328Hz...).

Subtraction: Artificially reduce the volume of those specific frequencies in the current spectrum array.

Repeat: Look for the next loudest peak in the remaining spectrum.

Stop when peaks are below the noise floor (Noise Gate).

Module D: Transient Detector (The "Speed" Layer)

FFT is accurate but slow. To feel "fast," we need to know when a pick hit the string, even if we don't know the pitch yet.

Time-Domain Analysis: Calculate the RMS (Volume) of the tiny incoming buffer (128 samples).

Flux Calculation: If volume spikes drastically compared to the previous buffer 
→
→
 Trigger "Note On" logic immediately.

Note: You might send the MIDI Note On command slightly after the pick detection once the FFT confirms the pitch (approx 3-5ms later).

Module E: MIDI Out (The Interface)

Library: mido using the rtmidi backend.

Logic:

State Machine: Keep track of currently playing notes.

Note Off Logic: If a frequency that was present in the last frame is missing in the current frame, send NOTE_OFF.

Velocity: Map the FFT Magnitude (Volume) to MIDI Velocity (0-127).

4. Required Tech Stack (Requirements.txt)

Give this list to the developers.

code
Text
download
content_copy
expand_less
numpy>=1.24.0      # For array manipulation
numba>=0.57.0      # For JIT compiling the math to C-speed
sounddevice>=0.4.6 # For ASIO Input (requires ASIO SDK on system)
mido>=1.2.10       # For MIDI message creation
python-rtmidi>=1.4 # Backend for Mido
scipy>=1.10.0      # Optional, for advanced signal windows
5. Developer Workflow / "First Sprint" Tasks

Task 1: The ASIO Loop

Create a script that opens the Behringer UM2 ASIO driver using sounddevice.

Pass audio through to the speakers.

Goal: Verify Python can hold the stream open without crashing or crackling at 128 buffer size.

Task 2: The Visualizer (Console)

Implement the Numba FFT.

Print the "Dominant Frequency" to the console (Monophonic test).

Goal: Pluck E string, see "82.4 Hz" print to console instantly.

Task 3: The Polyphonic Logic

Implement the "Iterative Spectral Subtraction" described in Module C.

Print list of frequencies to console.

Goal: Strum an E-Major chord, see [82, 123, 164, 207, 246, 329] (approx) in console.

Task 4: The Rust Connection

Connect the logic to mido.

Output to loopMIDI.

Goal: Play Rust.

6. Pitfalls to Avoid (Advice for the Team)

Do not use Matplotlib for debugging: It is too slow for real-time. Use console text bars ||||||| or a dedicated fast GUI like pyqtgraph (only for debugging).

Garbage Collection: Pre-allocate all Numpy arrays. Do not create new arrays inside the audio callback loop. Python's Garbage Collector will cause lag spikes if you create variables every millisecond.

44.1k vs 48k: Hardcode the app to 48,000Hz to match the Behringer native clock and Rust. Do not allow resampling.
