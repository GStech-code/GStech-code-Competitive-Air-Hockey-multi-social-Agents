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
        self.x_adv = 0
        self.y_adv = 0
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
                raise Exception('Queue is full')
            self._dq.append(cmd)
        self.x_adv += cmd[0]
        self.y_adv += cmd[1]

    def push_multiple(self, cmds: List[Command]) -> None:
        length_new = len(cmds)
        with self._lock:
            if len(self._dq) + length_new > self._cap:
                raise Exception('Inserting too many objects')
            self._dq.extend(cmds)
        for cmd in cmds:
            self.x_adv += cmd[0]
            self.y_adv += cmd[1]

    def pop(self) -> Optional[Command]:
        with self._lock:
            if not self._dq:
                return None
            cmd = self._dq.popleft()
            self.x_adv -= cmd[0]
            self.y_adv -= cmd[1]
            return cmd

    def flush_newest(self, k: int) -> None:
        with self._lock:
            for _ in range(min(k, len(self._dq))):
                cmd = self._dq.pop()
                self.x_adv -= cmd[0]
                self.y_adv -= cmd[1]

    def clear(self) -> None:
        with self._lock:
            self._dq.clear()
        self.x_adv = 0
        self.y_adv = 0

class Mailbox:
    """Wiring for your semantics."""
    def __init__(self, cmd_capacity: int = 32):
        self.latest_world_state = LatestVal()
        self.latest_instruction = LatestVal()
        self.status_change_flag = LatestChangeStatus()
        self.commands = STCommandQueue(cmd_capacity)
        self.shutdown = threading.Event()
