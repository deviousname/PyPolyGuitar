import numpy as np

class RingBuffer:
    def __init__(self, capacity, dtype=np.float32):
        self.capacity = capacity
        self.dtype = dtype
        self.buffer = np.zeros(capacity, dtype=dtype)
        self.write_index = 0
        self.is_full = False

    def write(self, data):
        """
        Writes data to the ring buffer.
        """
        data_len = len(data)
        if data_len > self.capacity:
            # If data is larger than buffer, just take the last capacity samples
            data = data[-self.capacity:]
            data_len = self.capacity

        # Calculate indices
        end_index = self.write_index + data_len

        if end_index <= self.capacity:
            # Simple write (no wrap-around needed for this chunk)
            self.buffer[self.write_index:end_index] = data
            self.write_index = end_index
        else:
            # Wrap-around write
            first_part_len = self.capacity - self.write_index
            self.buffer[self.write_index:] = data[:first_part_len]
            second_part_len = data_len - first_part_len
            self.buffer[:second_part_len] = data[first_part_len:]
            self.write_index = second_part_len
            self.is_full = True

        if self.write_index == self.capacity:
            self.write_index = 0
            self.is_full = True

    def read_recent(self, n_samples):
        """
        Returns the most recent n_samples from the buffer.
        """
        if n_samples > self.capacity:
            raise ValueError("Requested samples exceed buffer capacity")

        if self.write_index >= n_samples:
            return self.buffer[self.write_index - n_samples : self.write_index].copy()
        else:
            # Need to wrap around to get the most recent data
            part1_len = n_samples - self.write_index
            part1 = self.buffer[-part1_len:]
            part2 = self.buffer[:self.write_index]
            return np.concatenate((part1, part2))
