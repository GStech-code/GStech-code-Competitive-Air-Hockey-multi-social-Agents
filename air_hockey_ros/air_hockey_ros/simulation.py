import logging
from typing import Dict
logging.basicConfig(filename='game_manager.log', level=logging.INFO)

class Simulation:
    def __init__(self, **params):
        self.width = params.get('width', 0)
        self.height = params.get('height', 0)
        self.step_size = params.get('step_size', 1)

    def reset_game(self, num_agents_team_a, num_agents_team_b):
        self.team_a_score = 0
        self.team_b_score = 0
        self.num_agents_team_a = num_agents_team_a
        self.num_agents_team_b = num_agents_team_b
        self.num_agents = num_agents_team_a + num_agents_team_b

    def end_game(self):
        return {"team_a_score": self.team_a_score, "team_b_score": self.team_b_score}

    def apply_commands(self, commands):
        raise NotImplementedError('This method needs to be implemented')

    def get_world_state(self) -> Dict:
        """
        Notice: when implementing, return a copy of world state.
        """
        raise NotImplementedError('This method needs to be implemented')