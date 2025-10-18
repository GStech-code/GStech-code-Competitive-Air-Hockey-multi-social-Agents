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
    def __init__(self, capacity: int = 32, min_capacity: int = 1):
        self._dq = deque(maxlen=capacity)
        self.x_adv = 0
        self.y_adv = 0
        self._cap = capacity
        self._min_cap = min_capacity
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
    def get_advance(self) -> Tuple[int, int]:
        with self._lock:
            return self.x_adv, self.y_adv

    def push(self, cmd: Command) -> None:
        with self._lock:
            if len(self._dq) >= self._cap:
                raise Exception('Queue is full')
            self._dq.append(cmd)
            self.x_adv += cmd[0]
            self.y_adv += cmd[1]

    def push_multiple(self, cmds: List[Command]) -> None:
        length_new = len(cmds)
        sx, sy = 0, 0
        for cmd in cmds:
            sx += cmd[0]
            sy += cmd[1]
        with self._lock:
            if len(self._dq) + length_new > self._cap:
                raise Exception('Inserting too many objects')
            self._dq.extend(cmds)
            self.x_adv += sx
            self.y_adv += sy

    def pop(self) -> Optional[Command]:
        with self._lock:
            if not self._dq:
                return None
            cmd = self._dq.popleft()
            self.x_adv -= cmd[0]
            self.y_adv -= cmd[1]
            return cmd

    def peek_all(self) -> List[Command]:
        with self._lock:
            return [cmd for cmd in self._dq]

    def flush_newest(self, k: int) -> None:
        popped = 0
        sx, sy = 0, 0
        dq = self._dq
        with self._lock:
            while dq and popped < k:
                cmd = dq.pop()
                sx += cmd[0]
                sy += cmd[1]
                popped += 1
            self.x_adv -= sx
            self.y_adv -= sy

    def flush_limit(self) -> None:
        sx, sy = 0, 0
        dq = self._dq
        with self._lock:
            current = len(dq)
            while current > self._min_cap:
                cmd = dq.pop()
                sx += cmd[0]
                sy += cmd[1]
                current -= 1
            self.x_adv -= sx
            self.y_adv -= sy

    def clear(self) -> None:
        with self._lock:
            self._dq.clear()
            self.x_adv = 0
            self.y_adv = 0

class Mailbox:
    """Wiring for your semantics."""
    def __init__(self, cmd_capacity: int = 32, cmd_min_capacity=2):
        self.latest_frame = LatestVal()
        self.latest_instruction = LatestVal()
        self.status_change_flag = LatestChangeStatus()
        self.commands = STCommandQueue(cmd_capacity, cmd_min_capacity)
        self.shutdown = threading.Event()
