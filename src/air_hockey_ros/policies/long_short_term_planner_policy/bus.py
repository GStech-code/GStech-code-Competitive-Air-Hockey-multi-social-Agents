import threading
from collections import deque
from typing import Optional, Tuple, List

Command = Tuple[int, int]

class LatestVal:
    def __init__(self):
        self._val: Optional[object] = None
    def set(self, v):
        self._val = v
    def get(self):
        return self._val

class LatestChangeStatus:
    def __init__(self):
        self._status = False
    def inform(self):
        self._status = True

    def acknowledge(self) -> bool:
        status = self._status
        self._status = False
        return status

class STCommandQueue:
    def __init__(self, capacity: int = 32):
        self._dq = deque(maxlen=capacity)
        self._cap = capacity
        self._lock = threading.Lock()

    def get_capacity(self) -> int:
        return self._cap

    def get_size(self) -> int:
        with self._lock:
            return len(self._dq)

    def is_full(self) -> bool:
        with self._lock:
            return len(self._dq) == self._cap
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._dq) == 0

    def push(self, cmd: Command) -> None:
        with self._lock:
            if len(self._dq) >= self._cap:
                # drop NEWEST first to preserve items awaiting pop
                self._dq.pop()
            self._dq.append(cmd)  # newest on right

    def push_multiple(self, cmds: List[Command]) -> None:
        length_new = len(cmds)
        with self._lock:
            overrides = length_new + len(self._dq) - self._cap
            for i in range(overrides):
                self._dq.pop()
            self._dq.extend(cmds)

    def pop(self) -> Optional[Command]:
        with self._lock:
            if not self._dq:
                return None
            return self._dq.popleft()  # oldest first

    def flush_newest(self, k: int) -> None:
        with self._lock:
            for _ in range(min(k, len(self._dq))):
                self._dq.pop()

    def clear(self) -> None:
        with self._lock:
            self._dq.clear()
class Mailbox:
    """Wiring for your semantics."""
    def __init__(self, cmd_capacity: int = 32):
        self.latest_world_state = LatestVal()
        self.latest_instruction = LatestVal()
        self.status_change_flag = LatestChangeStatus()
        self.commands = STCommandQueue(cmd_capacity)
        self.shutdown = threading.Event()
