from typing import Dict
from .types import Command
from air_hockey_ros import AgentPolicy
from .short_term_objectives import Objective, OBJECTIVES, ObjectiveEnum

class ShortAgentPolicy(AgentPolicy):
    def __init__(self, agent_id, teammate_ids, objective_enum: ObjectiveEnum,
                 rules: Dict = None, obj_params: Dict = None):
        super().__init__(agent_id)
        self.objectiveEnum = objective_enum
        rules = rules if rules else {}
        obj_params = obj_params if obj_params else {}
        self.objective: Objective = OBJECTIVES[self.objectiveEnum](agent_id, teammate_ids, rules, **obj_params)
    def update(self, world_state: Dict) -> Command:
        return self.objective.step(world_state)
