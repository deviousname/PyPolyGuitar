import mido
import math

class MidiInterface:
    def __init__(self, port_name="loopMIDI Port"):
        self.port_name = port_name
        self.output_port = None

        # State tracking
        self.active_midi_notes = set() # Set of MIDI numbers currently ON

        # Debounce tracking: {midi_note: frames_missing_count}
        self.missing_counter = {}
        self.FRAMES_TO_KILL = 3 # Note must be missing for 3 frames to turn off

    def open_port(self):
        try:
            self.output_port = mido.open_output(self.port_name)
            print(f"MIDI Port Opened: {self.port_name}")
        except:
            print("Could not open MIDI port. Is loopMIDI running?")

    def close_port(self):
        if self.output_port:
            for note in list(self.active_midi_notes):
                self.send_note_off(note)
            self.output_port.close()

    def send_note_on(self, note, vel=100):
        if self.output_port:
            self.output_port.send(mido.Message('note_on', note=note, velocity=vel))

    def send_note_off(self, note):
        if self.output_port:
            self.output_port.send(mido.Message('note_off', note=note, velocity=0))

    def update_notes(self, detected_freqs):
        if not self.output_port: return

        # 1. Convert new freqs to MIDI note numbers
        incoming_midi_notes = set()
        for f in detected_freqs:
            if f > 60: # Ignore rumble below ~60Hz
                # Formula: 69 + 12 * log2(freq / 440)
                midi_num = int(round(69 + 12 * math.log2(f / 440)))
                incoming_midi_notes.add(midi_num)

        # 2. Logic: New Notes (Turn ON)
        for note in incoming_midi_notes:
            if note not in self.active_midi_notes:
                self.send_note_on(note)
                self.active_midi_notes.add(note)

            # Reset missing counter if note is present
            if note in self.missing_counter:
                del self.missing_counter[note]

        # 3. Logic: Missing Notes (Debounce OFF)
        # Check notes that are currently active but NOT in incoming
        candidates_for_off = self.active_midi_notes - incoming_midi_notes

        for note in candidates_for_off:
            # Increment missing counter
            self.missing_counter[note] = self.missing_counter.get(note, 0) + 1

            # If missing for enough frames, kill it
            if self.missing_counter[note] >= self.FRAMES_TO_KILL:
                self.send_note_off(note)
                self.active_midi_notes.remove(note)
                del self.missing_counter[note]
