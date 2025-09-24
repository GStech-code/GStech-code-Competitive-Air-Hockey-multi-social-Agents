from collections import deque
import math

class RollingAvgQueue:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.k = capacity
        self._dq = deque()
        self._sum = 0
        self._length = 0

    def __len__(self):
        return self._length

    def append(self, x):
        """Append x; if over capacity, drop oldest and update sum."""
        if self._length == self.k:
            self._sum -= self._dq.popleft()
        else:
            self._length += 1
        self._dq.append(x)
        self._sum += x

    def total(self):
        return self._sum

    def avg(self):
        if not self._dq:
            raise ZeroDivisionError("average of empty window")
        return self._sum / self._length

    # Optional helpers:
    def values(self):
        return tuple(self._dq)

    def clear(self):
        self._dq.clear()
        self._length = 0
        self._sum = 0

    def clear_and_re_capacity(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.k = capacity
        self.clear()


class RollingMinMaxQueue:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.k = capacity
        self._i = -1  # last pushed index
        self._dq_min = deque()  # (idx, val), nondecreasing by val
        self._dq_max = deque()  # (idx, val), nonincreasing by val

    def __len__(self):
        # number of items currently in the window
        if self._i < 0: return 0
        earliest = max(0, self._i - self.k + 1)
        return self._i - earliest + 1

    def append(self, x):
        """Append x; drops oldest if size would exceed capacity."""
        self._i += 1
        # maintain min deque (nondecreasing)
        while self._dq_min and self._dq_min[-1][1] >= x:
            self._dq_min.pop()
        self._dq_min.append((self._i, x))
        # maintain max deque (nonincreasing)
        while self._dq_max and self._dq_max[-1][1] <= x:
            self._dq_max.pop()
        self._dq_max.append((self._i, x))
        # expire anything outside the window
        expire_idx = self._i - self.k
        if self._dq_min and self._dq_min[0][0] <= expire_idx:
            self._dq_min.popleft()
        if self._dq_max and self._dq_max[0][0] <= expire_idx:
            self._dq_max.popleft()

    def min(self):
        if not self._dq_min: raise IndexError("empty")
        return self._dq_min[0][1]

    def max(self):
        if not self._dq_max: raise IndexError("empty")
        return self._dq_max[0][1]

    def range(self):
        """Return max(data) - min(data)."""
        return self.max() - self.min()

    def clear(self):
        self._dq_min.clear()
        self._dq_max.clear()
        self._i = -1

    def clear_and_re_capacity(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.k = capacity
        self.clear()
