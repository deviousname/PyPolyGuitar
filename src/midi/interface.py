# Project: PyPolyGuitar
# File: src/midi/interface.py

import mido
import time

class MidiInterface:
    def __init__(self, port_name="loopMIDI Port"):
        self.port_name = port_name
        self.output_port = None
        self.active_notes = {}  # Dictionary to track active notes: {note_number: velocity}

    def open_port(self):
        """
        Opens the MIDI output port.
        """
        try:
            available_ports = mido.get_output_names()
            print(f"Available MIDI Ports: {available_ports}")

            if self.port_name in available_ports:
                self.output_port = mido.open_output(self.port_name)
                print(f"Opened MIDI Port: {self.port_name}")
            else:
                # Fallback to the first available port or virtual port if supported
                if available_ports:
                    print(f"Port '{self.port_name}' not found. Opening '{available_ports[0]}'.")
                    self.output_port = mido.open_output(available_ports[0])
                else:
                    print("No MIDI ports found. MIDI output disabled.")
                    self.output_port = None
        except Exception as e:
            print(f"Error opening MIDI port: {e}")
            self.output_port = None

    def close_port(self):
        if self.output_port:
            self.output_port.close()
            print("MIDI Port closed.")

    def note_on(self, note, velocity=64):
        """
        Sends a Note On message.
        """
        if self.output_port and 0 <= note <= 127:
            msg = mido.Message('note_on', note=note, velocity=velocity)
            self.output_port.send(msg)
            self.active_notes[note] = velocity

    def note_off(self, note):
        """
        Sends a Note Off message.
        """
        if self.output_port and 0 <= note <= 127:
            msg = mido.Message('note_off', note=note, velocity=0)
            self.output_port.send(msg)
            if note in self.active_notes:
                del self.active_notes[note]

    def update_notes(self, detected_frequencies, min_velocity=60, max_velocity=100):
        """
        Updates the state of notes based on detected frequencies.
        :param detected_frequencies: List of frequencies in Hz.
        """
        # Convert frequencies to MIDI note numbers
        current_notes = set()
        for freq in detected_frequencies:
            if freq > 0:
                note = self._freq_to_midi(freq)
                current_notes.add(note)

        # Determine notes to turn off (active but not in current)
        notes_to_off = [n for n in self.active_notes if n not in current_notes]
        for note in notes_to_off:
            self.note_off(note)

        # Determine notes to turn on (current but not active)
        # For simplicity, we use a fixed velocity or map magnitude if available
        # Here we don't have magnitude for each freq yet, so we use a default
        for note in current_notes:
            if note not in self.active_notes:
                self.note_on(note, velocity=80)

    def _freq_to_midi(self, freq):
        """
        Converts frequency to MIDI note number.
        """
        # A4 = 440Hz = MIDI 69
        if freq <= 0:
            return 0
        import math
        # formula: 69 + 12 * log2(freq / 440)
        note = 69 + 12 * math.log2(freq / 440)
        return int(round(note))
