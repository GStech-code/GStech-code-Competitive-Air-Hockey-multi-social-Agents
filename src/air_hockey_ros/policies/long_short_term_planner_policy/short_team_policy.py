from typing import List
from .short_term_objectives import ObjectiveEnum
from .short_agent_policy import ShortAgentPolicy
from air_hockey_ros import TeamPolicy, register_policy, AgentPolicy

@register_policy('short')
class ShortTeamPolicy(TeamPolicy):
    def __init__(self, **params):
        super().__init__(**params)
        self.rules = params.copy()
        self.teammates_ids = [[id for id in self.agents_ids if id != current_agent]
                              for current_agent in self.agents_ids]
        for key in ('team', 'num_agents_team_a', 'num_agents_team_b'):
            self.rules.pop(key)
        goal_offset = self.rules.get('goal_offset', 40)
        paddle_radius = self.rules.get('paddle_radius', 20)
        unit_speed_px = self.rules.get('unit_speed_px', 4)

        self.objectives = [(ObjectiveEnum.FAST_SHOT, {}),
                           (ObjectiveEnum.INTERCEPT, {}),
                           (ObjectiveEnum.DEFEND_LINE, {}),
                           (ObjectiveEnum.PASS_SHOT, {}),
                           (ObjectiveEnum.DEFEND_LINE,
                            {'defense_row': 2 * (goal_offset + paddle_radius + unit_speed_px)})
                           ]
        n_agents = len(self.agents_ids)
        if n_agents < 5:
            self.objectives = self.objectives[:len(self.agents_ids)]
        elif n_agents > 5:
            for i in range(5, n_agents):
                self.objectives.append((ObjectiveEnum.PASS_SHOT, {}))

    def get_policies(self) -> List[AgentPolicy]:
        policies = [ShortAgentPolicy(agent_id=agent_id, teammate_ids=teammates, objective_enum=objective[0],
                                     rules=self.rules, obj_params=objective[1])
                    for agent_id, teammates, objective in zip(self.agents_ids, self.teammates_ids, self.objectives)]

        return policies
