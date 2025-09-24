from typing import List
from src.air_hockey_ros import TeamPolicy, register_policy, AgentPolicy
from .simple_regional_agent_policy import SimpleRegionalAgentPolicy
from .simple_crosser_agent_policy import SimpleCrosserAgentPolicy

@register_policy('simple')
class SimpleTeamPolicy(TeamPolicy):
    def __init__(self, **params):
        super().__init__(**params)
        self.width = params['width']
        self.height = params['height']
        self.goal_gap = params.get('goal_gap', 10)
        self.puck_radius = params.get('puck_radius', 1)
        self.paddle_radius = params.get('paddle_radius', 1)
        self.unit_speed_px = params.get('unit_speed_px', 4.0)

    def get_policies(self) -> List[AgentPolicy]:
        if len(self.agents_ids) == 0:
            return []
        x_cross = self.goal_gap + self.paddle_radius
        x_max_region = self.width//2 - self.paddle_radius
        y_min_region = self.paddle_radius
        y_max_region = self.height - self.paddle_radius
        if len(self.agents_ids) == 1:
            return [SimpleRegionalAgentPolicy(agent_id=self.agents_ids[0], x_min=x_cross, y_min=y_min_region,
                                            x_max=x_max_region, y_max=y_max_region, unit_speed_px=self.unit_speed_px)]
        x_min_region = x_cross + 2 * self.paddle_radius
        num_regions = len(self.agents_ids) - 1
        height_divided = self.height // num_regions
        region_confines = [i * height_divided for i in range(1, num_regions)]
        agent_min_ys = [y_min_region] + [confine - self.paddle_radius for confine in region_confines]
        agent_max_ys = [confine + self.paddle_radius for confine in region_confines] + [y_max_region]
        return ([SimpleRegionalAgentPolicy(agent_id=self.agents_ids[i], x_min=x_min_region, y_min=y_min,
                                           x_max=x_max_region, y_max=y_max, unit_speed_px=self.unit_speed_px,
                                           puck_radius=self.puck_radius, paddle_radius=self.paddle_radius,)
                 for i, (y_min, y_max) in enumerate(zip(agent_min_ys, agent_max_ys))]
                + [SimpleCrosserAgentPolicy(agent_id=self.agents_ids[-1],
                                            x_cross=x_cross, y_min=y_min_region, y_max=y_max_region)])
