from typing import Optional, Dict, Tuple
from bus import Mailbox

Command = Tuple[int, int]

# TODO: Proper implementation
class ShortTermPlanner:
    """
    Stateless-ish ST that advances in tiny steps.
    Called by SpotlightScheduler:
      - emergency_step  when in EMERGENCY spotlight
      - step()                during ST spotlight window
    """
    def __init__(self, mailbox: Mailbox):
        self.mailbox = mailbox
        self.emergency_func = None
        self.instruction: Optional[object] = None
        self.world_state: Optional[Dict] = None

    def set_emergency_func(self, emergency_func):
        self.emergency_func = emergency_func

    def step(self) -> None:
        self.world_state = self.mailbox.latest_world_state.get()
        change_flag = self.mailbox.status_change_flag.acknowledge()
        if change_flag:
            self.change_step()
        else:
            self.no_change_step()

    def change_step(self):
        self.instruction = self.mailbox.latest_instruction.get()
        self.mailbox.commands.push(self._compute_emergency_cmd())

    def no_change_step(self):
        if not self.world_state or self.mailbox.commands.is_full():
            return
        cmd = self._compute_cmd_fast()
        self.mailbox.commands.push(cmd)

    def emergency_step(self) -> None:
        self.world_state = self.mailbox.latest_world_state.get()
        change_flag = self.mailbox.status_change_flag.acknowledge()
        if change_flag:
            self.instruction = self.mailbox.latest_instruction.get()
        self.mailbox.commands.clear()
        self.mailbox.commands.push(self._compute_cmd_fast())

    def set_emergency(self):
        self.emergency_func()

    def _compute_cmd_fast(self) -> Command:
        # TODO: replace with your fast heuristic
        return 0, 0

    def _compute_emergency_cmd(self) -> Command:
        # TODO: replace with emergency corrective action
        return 0, 0
