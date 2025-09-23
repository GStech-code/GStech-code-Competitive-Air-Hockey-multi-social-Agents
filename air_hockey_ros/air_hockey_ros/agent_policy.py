from typing import Dict, Tuple
class AgentPolicy:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def update(self, world_state: Dict) -> Tuple[int, int]:
        raise NotImplementedError("This method needs to be implemented")