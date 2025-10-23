from typing import List
from air_hockey_ros import TeamPolicy, register_policy, AgentPolicy
from .neural_agent_ppo_policy import PPOAgentPolicy
import os
from pathlib import Path
import torch
from .load_weights import load_ac_state_dict_from_pkl

@register_policy('two_capped_neural')
class TwoCappedNeuralTeamPolicy(TeamPolicy):
    def __init__(self, **params):
        super().__init__(**params)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.device = 'cuda'
        self.width = params['width']
        self.height = params['height']
        paddle_radius = float(params.get('paddle_radius', 20))
        unit_speed_px = float(params.get('unit_speed_px', 4.0))
        self.half_line_distance = paddle_radius + unit_speed_px
        puck_max_speed = float(params.get('puck_max_speed', 6.0))
        self.max_speed = max(puck_max_speed, unit_speed_px) * 1.05
        self.teammates_ids = [[id for id in self.agents_ids if id != current_agent]
                              for current_agent in self.agents_ids]
        self.weights = self._load_weights()

        if self.team == 'A':
            self.opponents_ids = [id for id in range(self.num_agents_team_a,
                                                  self.num_agents_team_a + self.num_agents_team_b)]
        else:
            self.opponents_ids = [id for id in range(self.num_agents_team_a)]

    def _load_weights(self):
        base = Path(__file__).resolve().parent
        weights = []
        pkl_directory = 'two_cap_weights'
        for i in range(2):
            filename = f'ppo_agent_{i}.pkl'
            file_path = base / pkl_directory / filename
            weights.append(load_ac_state_dict_from_pkl(file_path))
        return weights

    def get_policies(self) -> List[AgentPolicy]:
        policies = [PPOAgentPolicy(agent_id=agent_id, device=self.device, width=self.width, height=self.height,
                                    half_line_distance=self.half_line_distance, max_speed=self.max_speed,
                                    teammate_ids=teammates, opponent_ids=self.opponents_ids,
                                    cap_agents_per_team=2, obs_dim=21, hidden_dim=128, action_dim=2,
                                    network_state_dict=weight)
                for agent_id, teammates, weight in zip(self.agents_ids, self.teammates_ids, self.weights)]

        weight_path = "neural_weights/weight.pt"
        if os.path.exists(weight_path):
            for policy in policies:
                policy.load(weight_path)
        return policies
