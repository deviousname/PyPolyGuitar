import unittest
from unittest.mock import MagicMock, patch
from src.midi.interface import MidiInterface

class TestMidiInterface(unittest.TestCase):
    def setUp(self):
        self.midi = MidiInterface(port_name="TestPort")
        self.midi.output_port = MagicMock()

    def test_freq_to_midi(self):
        # A4 = 440 -> 69
        self.assertEqual(self.midi._freq_to_midi(440), 69)
        # A2 = 110 -> 45
        self.assertEqual(self.midi._freq_to_midi(110), 45)
        # E2 = 82.41 -> 40
        self.assertEqual(self.midi._freq_to_midi(82.41), 40)

    def test_update_notes(self):
        # Initial state: empty
        detected_freqs = [440.0, 82.4] # A4, E2
        self.midi.update_notes(detected_freqs)

        # Check if note_on was called for 69 and 40
        # Check internal state
        self.assertIn(69, self.midi.active_notes)
        self.assertIn(40, self.midi.active_notes)

        # Next frame: only A4
        detected_freqs = [440.0]
        self.midi.update_notes(detected_freqs)

        # Check if note_off was called for 40
        self.assertIn(69, self.midi.active_notes)
        self.assertNotIn(40, self.midi.active_notes)

        # Output port check
        # We expect calls: note_on(69), note_on(40), note_off(40)
        # order depends on set iteration, but count should be right
        self.assertEqual(self.midi.output_port.send.call_count, 3)

if __name__ == '__main__':
    unittest.main()
