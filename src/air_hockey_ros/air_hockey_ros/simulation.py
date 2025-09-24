from typing import Dict, Tuple

class Simulation:
    def get_table_sizes(self) -> Tuple[int, int]:
        raise NotImplementedError('This method needs to be implemented')

    def end_game(self):
        raise NotImplementedError('This method needs to be implemented')

    def reset_game(self, num_agents_team_a, num_agents_team_b, **params):
        raise NotImplementedError('This method needs to be implemented')

    def apply_commands(self, commands):
        raise NotImplementedError('This method needs to be implemented')

    def get_world_state(self) -> Dict:
        """
        Notice: when implementing, return a copy of world state.
        """
        raise NotImplementedError('This method needs to be implemented')