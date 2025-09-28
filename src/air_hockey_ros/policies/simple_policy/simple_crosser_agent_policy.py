from typing import Tuple
import numpy as np
from air_hockey_ros import AgentPolicy

class SimpleCrosserAgentPolicy(AgentPolicy):
    def __init__(self, agent_id, x_cross, y_min, y_max, unit_speed_px):
        super().__init__(agent_id)
        half_unit_speed_px = unit_speed_px / 2
        self.x_min_cross = x_cross - half_unit_speed_px
        self.x_max_cross = x_cross + half_unit_speed_px
        self.y_min = y_min
        self.y_max = y_max
        self.y = 1

    def update(self, world_state) -> Tuple[int, int]:
        agent_x = world_state['agent_x'][self.agent_id]
        agent_y = world_state['agent_y'][self.agent_id]

        if agent_x > self.x_max_cross:
            x = -1
        elif agent_x < self.x_min_cross:
            x = 1
        else:
            x = 0
        if agent_y >= self.y_max:
            self.y = -1
        elif agent_y <= self.y_min:
            self.y = 1
        return x, self.y


