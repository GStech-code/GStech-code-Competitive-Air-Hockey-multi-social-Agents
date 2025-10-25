from typing import Dict
from .types import Command
from air_hockey_ros import AgentPolicy
from .short_term_objectives import Objective, OBJECTIVES, ObjectiveEnum

class ShortAgentPolicy(AgentPolicy):
    def __init__(self, agent_id,
                 objective_insert: Objective = None,
                 teammate_ids=None,
                 objective_enum: ObjectiveEnum = None,
                 rules: Dict = None, obj_params: Dict = None):
        super().__init__(agent_id)
        rules = rules if rules else {}
        obj_params = obj_params if obj_params else {}
        if objective_insert is not None:
            self.objective = objective_insert
            return
        else:
            if teammate_ids is None:
                teammate_ids = []
            if objective_enum is None:
                objective_enum = ObjectiveEnum.PASS_SHOT
            self.objective: Objective = OBJECTIVES[objective_enum](agent_id, teammate_ids, rules, **obj_params)
    def update(self, world_state: Dict) -> Command:
        return self.objective.step(world_state)
