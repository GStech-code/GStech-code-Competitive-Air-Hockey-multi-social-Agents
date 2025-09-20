from typing import List
from air_hockey_ros import TeamPolicy, register_policy, AgentPolicy
from .simple_regional_agent_policy import SimpleRegionalAgentPolicy
from .simple_crosser_agent_policy import SimpleCrosserAgentPolicy

@register_policy('simple_team')
class SimpleTeamPolicy(TeamPolicy):
    def __init__(self, **params):
        super().__init__(**params)
        self.width = params['width']
        self.height = params['height']
        self.goal_gap = params.get('goal_gap', 10)
        self.p_radius = params.get('p_radius', 1)
        self.step_size = params.get('step_size', 8)

    def get_policies(self) -> List[AgentPolicy]:
        if len(self.agents_ids) == 0:
            return []
        x_cross = self.goal_gap + self.p_radius
        x_max_region = self.width//2 - self.p_radius
        y_min_region = self.p_radius
        y_max_region = self.height//2 - self.p_radius
        if len(self.agents_ids) == 1:
            return [SimpleRegionalAgentPolicy(id=self.agents_ids[0], p_radius=self.p_radius,
                                             x_min=x_cross, y_min=y_min_region,
                                             x_max=x_max_region, y_max=y_max_region, step_size=self.step_size)]
        x_min_region = x_cross + 2 * self.p_radius
        num_regions = len(self.agents_ids) - 1
        height_divided = self.height // num_regions
        region_confines = [i * height_divided for i in range(1, num_regions)]
        agent_min_ys = [y_min_region] + [confine - self.p_radius for confine in region_confines]
        agent_max_ys = [confine + self.p_radius for confine in region_confines] + [y_max_region]
        return ([SimpleRegionalAgentPolicy(id=self.agents_ids[i], p_radius=self.p_radius,
                                             x_min=x_min_region, y_min=y_min,
                                             x_max=x_max_region, y_max=y_max, step_size=self.step_size)
                 for i, (y_min, y_max) in enumerate(zip(agent_min_ys, agent_max_ys))]
                + [SimpleCrosserAgentPolicy(id=self.agents_ids[-1], p_radius=self.p_radius,
                                            x_cross=x_cross, y_min=y_min_region, y_max=y_max_region)])
