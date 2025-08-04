class SimulationInterface:
    def __init__(self):
        pass

    def reset_game(self, **params):
        self.team_a_score = 0
        self.team_b_score = 0
        self.team_a_agents = params['team_a_agents']
        self.team_b_agents = params['team_b_agents']
        self.rules = params['rules']

    def end_game(self):
        return {"team_a_score": self.team_a_score, "team_b_score": self.team_b_score}

    def apply_commands(self, commands):
        raise NotImplementedError('This method needs to be implemented')

    def get_world_state(self) -> object:
        """
        Notice: when implementing, return a copy of world state.
        """
        raise NotImplementedError('This method needs to be implemented')