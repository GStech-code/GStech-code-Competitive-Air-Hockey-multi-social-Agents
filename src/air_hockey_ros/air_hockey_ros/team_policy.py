from typing import List
from .agent_policy import AgentPolicy

class TeamPolicy:
    def __init__(self, **params):
        self.num_agents_team_a = params['num_agents_team_a']
        self.num_agents_team_b = params['num_agents_team_b']
        self.team = params['team']
        if self.team == 'A':
            self.agents_ids = [id for id in range(self.num_agents_team_a)]
        else:
            self.agents_ids = [id for id in range(self.num_agents_team_a,
                                                  self.num_agents_team_a + self.num_agents_team_b)]

    def get_policies(self) -> List[AgentPolicy]:
        """
        Returns a list of policies for the objects.
        List length must be equal to the number of agents.
        """
        raise NotImplementedError("This method needs to be implemented")