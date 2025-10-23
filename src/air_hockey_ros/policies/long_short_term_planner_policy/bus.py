import threading
from typing import Optional, Tuple

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

class Mailbox:
    """Wiring for your semantics."""
    def __init__(self):
        self.latest_world_state = LatestVal()
        self.latest_instruction = LatestVal()
        self.status_change_flag = LatestChangeStatus()
        self.command = LatestVal()
        self.shutdown = threading.Event()
