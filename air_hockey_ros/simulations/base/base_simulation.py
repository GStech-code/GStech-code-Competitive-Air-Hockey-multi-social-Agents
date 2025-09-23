from __future__ import annotations
from typing import Dict, List, Tuple
from air_hockey_ros import Simulation, register_simulation
from .base_engine import BaseEngine

@register_simulation("base")
class BaseSimulation(Simulation):
    def __init__(self, **params):
        self.engine = BaseEngine(**params)

    def get_table_sizes(self) -> Tuple[int, int]:
        return self.engine.width, self.engine.height

    def end_game(self):
        return self.engine.get_scores()

    def reset_game(self, num_agents_team_a, num_agents_team_b, **params):
        self.engine.configure(**params)
        self.engine.reset(num_agents_team_a, num_agents_team_b,
                          params.get('agent_positions'),
                          params.get('puck_pos'),
                          params.get('puck_vel'))

    def apply_commands(self, commands: List[Tuple[int, int, int]]):
        self.engine.apply_commands(commands)
        self.engine.step()

    def get_world_state(self) -> Dict:
        return self.engine.world_state()
