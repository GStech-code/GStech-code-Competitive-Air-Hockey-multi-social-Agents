from typing import Dict, Tuple, Optional
from .bus import Mailbox
from .types import Command
from air_hockey_ros import AgentPolicy
from .short_term_objectives import Objective, OBJECTIVES, ObjectiveEnum

class ShortAgentPolicy(AgentPolicy):
    def __init__(self, agent_id, teammate_ids, objective_enum: ObjectiveEnum, commands_target: int = None,
                 cmd_capacity: int = 32, cmd_min_capacity: int = 2,
                 rules: Dict = None, obj_params: Dict = None):
        super().__init__(agent_id)
        self.teammate_ids = teammate_ids
        self.objectiveEnum = objective_enum
        self._mailbox: Optional[Mailbox] = None
        self._last_cmd: Command = (0, 0)
        self.update_counter = 0
        rules = rules if rules else {}
        obj_params = obj_params if obj_params else {}
        self.commands_max_target = commands_target if commands_target else (self.commands.get_capacity() // 4)
        self.mailbox = Mailbox(cmd_capacity=cmd_capacity,
                                cmd_min_capacity=cmd_min_capacity)
        self.commands = self.mailbox.commands
        self.objective: Objective = OBJECTIVES[self.objectiveEnum.name](self.agent_id, self.teammate_ids,
                                                                        self.commands, rules, **obj_params)
    def update(self, world_state: Dict) -> Command:
        if self.commands.get_size() < self.commands_max_target:
            self.objective.new_ws_step(world_state)
        return self.commands.pop()
