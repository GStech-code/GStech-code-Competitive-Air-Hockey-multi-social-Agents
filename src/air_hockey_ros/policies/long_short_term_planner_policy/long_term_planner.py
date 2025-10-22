import time
from typing import Optional, Dict, Tuple
from bus import Mailbox
Command = Tuple[int, int]

# TODO: Proper implementation
class LongTermPlanner:
    """
    LT advances in tiny chunks each scheduler turn.
    Can publish new objectives and request EMERGENCY spotlight if needed.
    """
    def __init__(self, mailbox: Mailbox):
        self.mailbox = mailbox
        self.world_state = None

    def step(self) -> None:
        if self.world_state is None:
            time.sleep(0.0005)
            return

    def change_instruction(self, instruction):
        self.mailbox.latest_instruction.set(instruction)
        self.mailbox.status_change_flag.inform()

