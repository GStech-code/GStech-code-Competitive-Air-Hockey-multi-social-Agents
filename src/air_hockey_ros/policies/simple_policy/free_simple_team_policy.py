from typing import List
from air_hockey_ros import TeamPolicy, register_policy, AgentPolicy
from .simple_regional_agent_policy import SimpleRegionalAgentPolicy
from .simple_crosser_agent_policy import SimpleCrosserAgentPolicy

@register_policy('free_simple')
class SimpleTeamPolicy(TeamPolicy):
    def __init__(self, **params):
        super().__init__(**params)
        self.width = params['width']
        self.height = params['height']
        self.goal_offset = params.get('goal_offset', 1)
        self.puck_radius = params.get('puck_radius', 1)
        self.paddle_radius = params.get('paddle_radius', 1)
        self.unit_speed_px = params.get('unit_speed_px', 4.0)

    def get_policies(self) -> List[AgentPolicy]:
        x_min_region = self.goal_offset + 2 * self.paddle_radius
        x_max_region = self.width // 2 - self.paddle_radius
        y_min_region = self.paddle_radius
        y_max_region = self.height - self.paddle_radius
        return [SimpleRegionalAgentPolicy(agent_id=agent_id, x_min=x_min_region, y_min=y_min_region,
                                           x_max=x_max_region, y_max=y_max_region, unit_speed_px=self.unit_speed_px,
                                           puck_radius=self.puck_radius, paddle_radius=self.paddle_radius)
                for agent_id in self.agents_ids]
