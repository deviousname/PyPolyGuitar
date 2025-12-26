import unittest
from unittest.mock import MagicMock
from src.midi.interface import MidiInterface

class TestMidiInterface(unittest.TestCase):
    def setUp(self):
        self.midi = MidiInterface(port_name="TestPort")
        self.midi.output_port = MagicMock()

    def test_update_notes(self):
        # Initial state: empty
        detected_freqs = [440.0, 82.4] # A4, E2
        self.midi.update_notes(detected_freqs)

        # Check internal state
        self.assertIn(69, self.midi.active_midi_notes)
        self.assertIn(40, self.midi.active_midi_notes)

        # Next frame: only A4
        # NOTE: With debounce logic, 82.4 (40) should NOT turn off immediately.
        # It needs to be missing for FRAMES_TO_KILL (3) frames.

        # Frame 1 missing
        detected_freqs = [440.0]
        self.midi.update_notes(detected_freqs)
        self.assertIn(40, self.midi.active_midi_notes)
        self.assertEqual(self.midi.missing_counter[40], 1)

        # Frame 2 missing
        self.midi.update_notes(detected_freqs)
        self.assertIn(40, self.midi.active_midi_notes)
        self.assertEqual(self.midi.missing_counter[40], 2)

        # Frame 3 missing -> Turn OFF
        self.midi.update_notes(detected_freqs)
        self.assertNotIn(40, self.midi.active_midi_notes)

        # Output port check
        # Calls:
        # 1. note_on(69)
        # 2. note_on(40)
        # 3. note_off(40) (after 3 frames)

        # Filter calls to note_on/off
        calls = self.midi.output_port.send.call_args_list
        # We expect at least 3 calls
        self.assertGreaterEqual(len(calls), 3)

if __name__ == '__main__':
    unittest.main()
