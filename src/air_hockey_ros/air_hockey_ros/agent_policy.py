from typing import Dict, Tuple
class AgentPolicy:
    def __init__(self, agent_id):
        """
        Notice that the initiated object is going to be pickled.
        Advised not to init threading at this point.
        """
        self.agent_id = agent_id

    def on_agent_init(self):
        """
        Lets the policy know the agent has re-pickled and dependency injected it.
        The game is about to start.
        Useful when using threading.
        """
        pass

    def update(self, world_state: Dict) -> Tuple[int, int]:
        raise NotImplementedError("This method needs to be implemented")
