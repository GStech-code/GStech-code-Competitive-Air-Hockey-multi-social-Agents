from typing import Dict, Tuple, List, Optional
from collections import deque
from .types import Command
from air_hockey_ros import AgentPolicy
from .short_term_objectives import Objective, OBJECTIVES, ObjectiveEnum

class STCommandQueueNotThreaded:
    def __init__(self, capacity: int = 32, min_capacity: int = 1):
        self._dq = deque(maxlen=capacity)
        self.x_adv = 0
        self.y_adv = 0
        self._cap = capacity
        self._min_cap = min_capacity

    def get_capacity(self) -> int:
        return self._cap

    def get_size(self) -> int:
        return len(self._dq)

    def is_full(self) -> bool:
        return len(self._dq) == self._cap
    def is_empty(self) -> bool:
        return len(self._dq) == 0
    def get_advance(self) -> Tuple[int, int]:
        return self.x_adv, self.y_adv

    def push(self, cmd: Command) -> None:
        self._dq.append(cmd)
        self.x_adv += cmd[0]
        self.y_adv += cmd[1]

    def push_multiple(self, cmds: List[Command]) -> None:
        self._dq.extend(cmds)
        self.x_adv += sx
        self.y_adv += sy

    def pop(self) -> Optional[Command]:
        if not self._dq:
            return None
        cmd = self._dq.popleft()
        self.x_adv -= cmd[0]
        self.y_adv -= cmd[1]
        return cmd

    def peek_all(self) -> List[Command]:
        return [cmd for cmd in self._dq]

    def flush_newest(self, k: int) -> None:
        popped = 0
        sx, sy = 0, 0
        dq = self._dq
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
        current = len(dq)
        while current > self._min_cap:
            cmd = dq.pop()
            sx += cmd[0]
            sy += cmd[1]
            current -= 1
        self.x_adv -= sx
        self.y_adv -= sy

    def clear(self) -> None:
        self._dq.clear()
        self.x_adv = 0
        self.y_adv = 0

class ShortAgentPolicy(AgentPolicy):
    def __init__(self, agent_id, teammate_ids, objective_enum: ObjectiveEnum, commands_target: int = None,
                 cmd_capacity: int = 32, cmd_min_capacity: int = 2,
                 rules: Dict = None, obj_params: Dict = None):
        super().__init__(agent_id)
        self.teammate_ids = teammate_ids
        self.objectiveEnum = objective_enum
        self.update_counter = 0
        rules = rules if rules else {}
        obj_params = obj_params if obj_params else {}
        self.commands = STCommandQueueNotThreaded(cmd_capacity, cmd_min_capacity)
        self.commands_max_target = commands_target if commands_target else (self.commands.get_capacity() // 4)
        self.objective: Objective = OBJECTIVES[self.objectiveEnum](self.agent_id, self.teammate_ids,
                                                                        self.commands, rules, **obj_params)
    def update(self, world_state: Dict) -> Command:
        if self.commands.get_size() < self.commands_max_target:
            self.objective.new_ws_step(world_state)
        return self.commands.pop()
