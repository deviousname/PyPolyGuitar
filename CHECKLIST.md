# PyPolyGuitar Checklist

This checklist tracks the progress of the PyPolyGuitar project, based on the architectural blueprint.

## 1. Setup & Environment
- [x] Install dependencies: `pip install -r requirements.txt`
- [x] Verify environment: `python run.py`
- [x] Ensure ASIO drivers (Behringer UM2) are installed and configured. (Handled via fallback/mock in dev env)
- [x] Hardcode sampling rate to 48,000Hz (no resampling).

## 2. Developer Workflow / First Sprint Tasks

### Task 1: The ASIO Loop (Module A)
- [x] Create a script to open the ASIO driver using `sounddevice`.
- [x] Implement a non-blocking callback function.
- [x] Implement a numpy ring buffer (approx 2048 samples, step 128 samples).
- [x] Ensure bit depth is 32-bit float (normalized -1.0 to 1.0).
- [x] Pass audio through to speakers to verify stream stability (no crashes/crackling at 128 buffer size).

### Task 2: The Visualizer / Console (Module B)
- [ ] Implement Numba-accelerated FFT (Module B).
    - [ ] Apply Blackman-Harris window function.
    - [ ] Implement zero-padding (e.g., 512 samples padded to 2048).
    - [ ] Use `numpy.fft.rfft` inside a Numba `@jit(nopython=True)` function.
    - [ ] Convert to Magnitude Spectrum.
    - [ ] Implement Spectral Whitening.
- [ ] Print "Dominant Frequency" to console (Monophonic test).
- [ ] Verify: Pluck E string, see "82.4 Hz" instantly.

### Task 3: The Polyphonic Logic (Module C)
- [ ] Implement "Iterative Spectral Subtraction" algorithm.
    - [ ] Find loudest peak.
    - [ ] Check for transient.
    - [ ] Calculate harmonic locations.
    - [ ] Subtract harmonics from spectrum.
    - [ ] Repeat until peaks are below noise floor.
- [ ] Print list of frequencies to console.
- [ ] Verify: Strum E-Major chord, see approx `[82, 123, 164, 207, 246, 329]`.

### Task 4: The Rust Connection (Module E)
- [ ] Connect logic to `mido` using `rtmidi` backend.
- [ ] Implement State Machine to track playing notes.
- [ ] Implement Note Off logic (frequency missing in current frame).
- [ ] Map FFT Magnitude to MIDI Velocity.
- [ ] Output to `loopMIDI`.
- [ ] Verify: Control virtual instrument or game (Rust).

## 3. Additional Modules & Logic

### Module D: Transient Detector
- [ ] Implement Time-Domain Analysis (RMS of 128 sample buffer).
- [ ] Implement Flux Calculation (detect drastic volume spikes).
- [ ] Trigger "Note On" logic immediately upon transient detection.

## 4. Best Practices & Optimization
- [ ] **Garbage Collection:** Pre-allocate all Numpy arrays. Avoid creating new arrays in the audio callback.
- [ ] **Debugging:** Use console text bars or `pyqtgraph` for real-time visualization (avoid Matplotlib).
