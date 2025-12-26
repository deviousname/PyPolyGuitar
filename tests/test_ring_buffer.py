import unittest
import numpy as np
from src.audio.ring_buffer import RingBuffer

class TestRingBuffer(unittest.TestCase):
    def test_write_and_read(self):
        rb = RingBuffer(10)
        data = np.arange(5, dtype=np.float32)
        rb.write(data)

        # Read back all 5
        read_back = rb.read_recent(5)
        np.testing.assert_array_equal(read_back, data)

        # Read back 3
        read_back_3 = rb.read_recent(3)
        np.testing.assert_array_equal(read_back_3, data[-3:])

    def test_wrap_around(self):
        rb = RingBuffer(5)
        # Write 3 samples: [0, 1, 2]
        rb.write(np.array([0, 1, 2], dtype=np.float32))
        # Write 3 more: [3, 4, 5]. Buffer should look like [5, 1, 2, 3, 4] effectively in circular terms
        # But specifically:
        # [0, 1, 2, 0, 0] -> write index 3
        # write [3, 4, 5]
        # [0, 1, 2] + [3, 4] (wrap after 2) -> buffer: [3, 4, 2, 3, 4]? No.

        # Initial: [0, 1, 2, _, _] (idx=3)
        # Write 3, 4, 5:
        # [3, 4, 5] -> needs 3 slots. available at end: 2.
        # Write 3, 4 at indices 3, 4. Buffer: [0, 1, 2, 3, 4]. idx wraps to 0.
        # Write 5 at index 0. Buffer: [5, 1, 2, 3, 4]. idx -> 1.

        rb.write(np.array([3, 4, 5], dtype=np.float32))

        # Most recent 5 should be 1, 2, 3, 4, 5
        read_back = rb.read_recent(5)
        np.testing.assert_array_equal(read_back, np.array([1, 2, 3, 4, 5], dtype=np.float32))

        # Most recent 3 should be 3, 4, 5
        read_back_3 = rb.read_recent(3)
        np.testing.assert_array_equal(read_back_3, np.array([3, 4, 5], dtype=np.float32))

    def test_overflow_write(self):
        rb = RingBuffer(5)
        data = np.arange(10, dtype=np.float32) # 0..9
        rb.write(data)

        # Should contain last 5: 5, 6, 7, 8, 9
        read_back = rb.read_recent(5)
        np.testing.assert_array_equal(read_back, np.array([5, 6, 7, 8, 9], dtype=np.float32))

if __name__ == '__main__':
    unittest.main()
