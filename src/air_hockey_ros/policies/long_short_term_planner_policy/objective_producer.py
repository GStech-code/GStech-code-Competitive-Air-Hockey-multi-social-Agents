# objective_producer.py
from typing import List, Dict, Tuple
from .short_term_objectives import Objective, ObjectiveEnum, OBJECTIVES_DICT
from .short_agent_policy import ShortAgentPolicy

class ObjectivesProducer:
    def __init__(self, agents_ids: List[int], num_valid_agents: int, teammate_ids: List[List[int]], **rules):
        self.agent_ids = agents_ids
        self.num_agents = len(agents_ids)
        self.num_valid_agents = num_valid_agents
        self.teammate_ids = teammate_ids
        self.rules = rules
        width = rules.get('width', 800)
        half_line = width * 0.5
        quarter_line = width * 0.25
        goal_offset = rules.get('goal_offset', 40)
        paddle_radius = rules.get('paddle_radius', 20)
        unit_speed_px = rules.get('unit_speed_px', 4)
        x_limit = half_line - paddle_radius - unit_speed_px

        start_x_home = max(quarter_line, goal_offset + paddle_radius + unit_speed_px)
        self.defense_rows = [min(x_limit, i * (goal_offset + paddle_radius + unit_speed_px))
                        for i in range(1, self.num_valid_agents + 1)]
        self.x_homes = [min(x_limit, start_x_home + i * (goal_offset + paddle_radius + unit_speed_px))
                   for i in range(self.num_valid_agents)]
        self.enforcement_lines = [width * (0.375 + i * 0.25) for i in range(self.num_valid_agents)]

    def produce_valid_objectives(self) -> Dict[Tuple[int, ObjectiveEnum], Objective]:
        defense_objectives = [OBJECTIVES_DICT[ObjectiveEnum.DEFEND_LINE](agent_id=self.agent_ids[i],
                                                                         teammate_ids=self.teammate_ids[i],
                                                                         rules=self.rules,
                                                                         defense_row=self.defense_rows[i])
                              for i in range(self.num_valid_agents)]

        intercept_objectives = [OBJECTIVES_DICT[ObjectiveEnum.INTERCEPT](agent_id=self.agent_ids[i],
                                                                         teammate_ids=self.teammate_ids[i],
                                                                         rules=self.rules,
                                                                         x_home=self.x_homes[i])
                                for i in range(self.num_valid_agents)]

        pass_shot_objectives = [OBJECTIVES_DICT[ObjectiveEnum.PASS_SHOT](agent_id=self.agent_ids[i],
                                                                         teammate_ids=self.teammate_ids[i],
                                                                         rules=self.rules)
                                for i in range(self.num_valid_agents)]
        fast_shot_objectives = [OBJECTIVES_DICT[ObjectiveEnum.FAST_SHOT](agent_id=self.agent_ids[i],
                                                                         teammate_ids=self.teammate_ids[i],
                                                                         rules=self.rules,
                                                                         enforcement_line=self.enforcement_lines[i])
                                for i in range(self.num_valid_agents)]

        objectives = {}
        for i, aid in enumerate(self.agent_ids[:self.num_valid_agents]):
            objectives[(aid, ObjectiveEnum.DEFEND_LINE)] = defense_objectives[i]
            objectives[(aid, ObjectiveEnum.INTERCEPT)] = intercept_objectives[i]
            objectives[(aid, ObjectiveEnum.PASS_SHOT)] = pass_shot_objectives[i]
            objectives[(aid, ObjectiveEnum.FAST_SHOT)] = fast_shot_objectives[i]

        return objectives

    def produce_pass_objectives(self) -> Dict[int, Objective]:
        return {self.agent_ids[i]: OBJECTIVES_DICT[ObjectiveEnum.PASS_SHOT](
            agent_id=self.agent_ids[i],
            teammate_ids=self.teammate_ids[i],
            rules=self.rules)
            for i in range(self.num_valid_agents, self.num_agents)}
