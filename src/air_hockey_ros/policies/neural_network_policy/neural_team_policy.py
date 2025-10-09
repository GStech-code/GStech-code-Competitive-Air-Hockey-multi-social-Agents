from typing import List
from air_hockey_ros import TeamPolicy, register_policy, AgentPolicy
from .neural_agent_policy import NeuralAgentPolicy
import os
import torch

@register_policy('neural')
class NeuralTeamPolicy(TeamPolicy):
    def __init__(self, **params):
        super().__init__(**params)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.device = 'cuda'
        self.width = params['width']
        self.height = params['height']
        puck_max_speed = float(params.get('puck_max_speed', 6.0))
        unit_speed_px = float(params.get('unit_speed_px', 4.0))
        self.max_speed = max(puck_max_speed, unit_speed_px) * 1.05
        self.teammates_ids = [[id for id in self.agents_ids if id != current_agent]
                              for current_agent in self.agents_ids]

        if self.team == 'A':
            self.opponents_ids = [id for id in range(self.num_agents_team_a,
                                                  self.num_agents_team_a + self.num_agents_team_b)]
        else:
            self.opponents_ids = [id for id in range(self.num_agents_team_a)]

    def get_policies(self) -> List[AgentPolicy]:
        policies =  [NeuralAgentPolicy(agent_id=agent_id, device=self.device, width=self.width, height=self.height,
                                  max_speed=self.max_speed, teammate_ids=teammates, opponent_ids=self.opponents_ids)
                for agent_id, teammates in zip(self.agents_ids, self.teammates_ids)]

        weight_path = "neural_weights/weight.pt"
        if os.path.exists(weight_path):
            for policy in policies:
                policy.load(weight_path)
        return policies
