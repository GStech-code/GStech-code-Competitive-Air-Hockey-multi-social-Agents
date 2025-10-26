# ls_team_policy.py
from typing import List
from .short_term_objectives import ObjectiveEnum, OBJECTIVE_COMBINATIONS
from .ls_agent_policy import LSAgentPolicy
from .short_agent_policy import ShortAgentPolicy
from air_hockey_ros import TeamPolicy, register_policy, AgentPolicy

VALID_AGENTS = 4
@register_policy('long_short')
class LongShortTeamPolicy(TeamPolicy):
    def __init__(self, **params):
        super().__init__(**params)
        self.rules = params.copy() if params else {}
        teammates_ids = [[id for id in self.agents_ids if id != current_agent]
                              for current_agent in self.agents_ids]
        for key in ('team', 'num_agents_team_a', 'num_agents_team_b'):
            if key in params:
                self.rules.pop(key)
        num_agents = len(self.agents_ids)
        self.num_valid_agents = min(num_agents, VALID_AGENTS)

        self.valid_agents = self.agents_ids[:self.num_valid_agents]

        if self.num_valid_agents > 0:
            object_combinations = OBJECTIVE_COMBINATIONS[self.num_valid_agents][1]
            self.starter_objectives = {id: obj for id, obj in zip(self.agents_ids, object_combinations)}
        else:
            self.starter_objectives = {}

        if num_agents > VALID_AGENTS:
            self.pass_agents = self.agents_ids[VALID_AGENTS:]
            self.pass_teammates = teammates_ids[VALID_AGENTS:]
        else:
            self.pass_agents = []
            self.pass_teammates = []

    def get_policies(self) -> List[AgentPolicy]:
        policies = [LSAgentPolicy(agent_id=agent_id, team=self.team,
                                  max_num_valid_agents=VALID_AGENTS,
                                  num_agents_team_a=self.num_agents_team_a,
                                  num_agents_team_b=self.num_agents_team_b,
                                  rules=self.rules,
                                  starter_objective=self.starter_objectives[agent_id]
                                  ) for agent_id in self.valid_agents]

        policies.extend([ShortAgentPolicy(agent_id=agent_id, teammate_ids=teammate_ids,
                                          objective_enum=ObjectiveEnum.PASS_SHOT, rules=self.rules)
                         for agent_id, teammate_ids in zip(self.pass_agents, self.pass_teammates)])
        return policies
