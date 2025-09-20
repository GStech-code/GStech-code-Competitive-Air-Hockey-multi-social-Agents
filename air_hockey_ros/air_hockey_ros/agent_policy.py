from typing import Dict, Tuple
class AgentPolicy:
    def __init__(self, id, p_radius):
        self.id = id
        self.p_radius = p_radius
    def update(self, world_state: Dict) -> Tuple[int, int]:
        raise NotImplementedError("This method needs to be implemented")