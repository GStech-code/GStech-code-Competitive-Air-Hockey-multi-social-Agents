from typing import List

class TeamPolicy:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents

    def get_policies(self) -> List[object]:
        """
        Returns a list of policies for the objects.
        List length must be equal to the number of agents.
        """
        raise NotImplementedError("This method needs to be implemented")