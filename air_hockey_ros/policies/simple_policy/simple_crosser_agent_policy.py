from typing import Tuple
import numpy as np
from air_hockey_ros import AgentPolicy

def norm_down_num(num):
    abs_num = np.abs(num)
    if abs_num < 1:
        return 0
    return int(num/abs_num)

class SimpleCrosserAgentPolicy(AgentPolicy):
    def __init__(self, id, p_radius, x_cross, y_min, y_max):
        super().__init__(id, p_radius)
        self.x_cross = x_cross
        self.y_min = y_min
        self.y_max = y_max
        self.y = 1

    def update(self, world_state) -> Tuple[int, int]:
        agent_x = world_state['agent_x'][self.id]
        agent_y = world_state['agent_y'][self.id]


        x = norm_down_num(self.x_cross - agent_x)
        if agent_y >= self.y_max:
            self.y = -1
        elif agent_y <= self.y_min:
            self.y = 1
        return x, self.y


