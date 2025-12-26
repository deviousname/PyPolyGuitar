from numba import jit
from numba.typed import List
import numpy as np

@jit(nopython=True)
def test_clear(l):
    # Try to clear
    # del l[:] is not supported in nopython mode for typed list usually
    # pop loop?
    while len(l) > 0:
        l.pop()
    l.append(1.0)
    return l

l = List()
l.append(10.0)
print(f"Before: {l}")
test_clear(l)
print(f"After: {l}")
