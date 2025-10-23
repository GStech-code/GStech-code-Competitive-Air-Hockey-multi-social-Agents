from typing import Optional, Dict, Tuple, Callable

class Objective:
    def __init__(self, agent_id, teammate_ids, rules: Dict, **params):
        self.agent_id = agent_id
        self.teammate_ids = teammate_ids
        self.rules = rules

    def step(self, ws: Dict) -> Tuple[int, int]:
        return 0, 0
