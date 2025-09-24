from __future__ import annotations
from typing import Dict, List, Tuple
from src.air_hockey_ros import Simulation, register_simulation
from .base_engine import BaseEngine
from .world_state_view import PygameView

def noops():
    pass

@register_simulation("base")
class BaseSimulation(Simulation):
    def __init__(self, view: bool=False):
        self.engine = BaseEngine()
        if view:
            self._py_view = PygameView()
            self._after_step = self._draw
        else:
            self._py_view = None
            self._after_step = noops

    def get_table_sizes(self) -> Tuple[int, int]:
        return self.engine.width, self.engine.height

    def end_game(self):
        if self._py_view:
            self._py_view.close()
        return self.engine.get_scores()

    def get_world_state(self) -> Dict:
        return self.engine.get_world_state()

    def _draw(self):
        self._py_view.draw(self.engine.get_world_state())

    def reset_game(self, num_agents_team_a, num_agents_team_b, **params):
        self.engine.configure(**params)
        self.engine.reset(num_agents_team_a, num_agents_team_b,
                          params.get('agent_positions'),
                          params.get('puck_pos'),
                          params.get('puck_vel'))
        if self._py_view is not None:
            self._py_view.reset(num_agents_team_a=num_agents_team_a, num_agents_team_b=num_agents_team_b, **params)

    def apply_commands(self, commands: List[Tuple[int, int, int]]):
        self.engine.apply_commands(commands)
        self.engine.step()
        self._after_step()
