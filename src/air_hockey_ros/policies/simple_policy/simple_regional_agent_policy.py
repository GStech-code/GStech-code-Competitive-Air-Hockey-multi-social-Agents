from typing import Tuple
import numpy as np
from air_hockey_ros import AgentPolicy

def norm_down_num(num):
    abs_num = np.abs(num)
    if abs_num < 1:
        return 0
    return int(num/abs_num)

class SimpleRegionalAgentPolicy(AgentPolicy):
    def __init__(self, agent_id, puck_radius, paddle_radius, x_min, x_max, y_min, y_max, unit_speed_px):
        super().__init__(agent_id)
        self.paddle_margin = puck_radius + paddle_radius
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.unit_speed_px = unit_speed_px

    def update(self, world_state) -> Tuple[int, int]:
        agent_x = world_state['agent_x'][self.agent_id]
        agent_y = world_state['agent_y'][self.agent_id]
        puck_x =world_state['puck_x']
        puck_y = world_state['puck_y']
        puck_vx = world_state['puck_vx']
        puck_vy = world_state['puck_vy']

        valid_x = False
        valid_y = False
        if agent_x < self.x_min:
            x = 1
        elif agent_x > self.x_max:
            x = -1
        else:
            valid_x = True

        if agent_y < self.y_min:
            y = 1
        elif agent_y > self.y_max:
            y = -1
        else:
            valid_y = True

        if valid_x and valid_y:
            if agent_x <= puck_x - self.paddle_margin:
                return self.up_puck_action(agent_x, agent_y, puck_x, puck_y, puck_vx, puck_vy)
            else:
                return self.down_puck_action(agent_x, agent_y, puck_x, puck_y)

        x_diff = puck_x - agent_x
        y_diff = puck_y - agent_y
        y_diff_abs = np.abs(y_diff)
        if x_diff < 0 and x_diff > -self.paddle_margin and y_diff_abs < self.paddle_margin:
            if y_diff == 0:
                y = 1
            else:
                y = -int(y_diff/y_diff_abs)
            return 1, y
        if valid_x:
            x = norm_down_num(x_diff)
            if ((x == 1 and agent_x >= self.x_max - self.unit_speed_px) or
                    (x == -1 and agent_x <= self.x_min + self.unit_speed_px)):
                return 0, y
        if valid_y:
            y = norm_down_num(y_diff)
            if ((y == 1 and agent_y <= self.y_max - self.unit_speed_px) or
                    (y ==-1 and agent_y >= self.y_min + self.unit_speed_px)):
                return x, 0
        return x, y

    def up_puck_action(self, agent_x, agent_y, puck_x, puck_y, puck_vx, puck_vy,
                      favor_behind=True, behind_bias=1.0):
        best = None
        best_score = float("inf")

        rx0 = agent_x - (puck_x - self.paddle_margin)
        ry0 = agent_y - (puck_y - self.paddle_margin)

        agent_possible_vx = [(0, 0)]
        if agent_x != self.x_min:
            agent_possible_vx.append((-1, -self.unit_speed_px))
        if agent_x != self.x_max:
            agent_possible_vx.append((1, self.unit_speed_px))
        agent_possible_vy = [(0, 0)]
        if agent_y != self.y_min:
            agent_possible_vy.append((-1, -self.unit_speed_px))
        if agent_y != self.y_max:
            agent_possible_vy.append((1, self.unit_speed_px))

        for agent_x_cmd, agent_vx in agent_possible_vx:
            for agent_y_cmd, agent_vy in agent_possible_vy:
                vx = agent_vx - puck_vx
                vy = agent_vy - puck_vy
                v2 = vx * vx + vy * vy

                # Closed-form best time (clamped to t>=0)
                if v2 > 0:
                    t_star = -(vx * rx0 + vy * ry0) / v2
                    if t_star <= 0:
                        t_star = 0.0  # evaluate at now (0+)
                else:
                    t_star = 0.0  # relative velocity zero -> distance constant

                # Distance at t_star to the puck's back
                dx = (rx0 + vx * t_star)
                dy = (ry0 + vy * t_star)
                dist2 = dx * dx + dy * dy

                # Optional soft bias: encourage "strike from behind" if feasible
                if favor_behind:
                    behind_now = agent_x <= (puck_x - self.paddle_margin)
                    closing_x = vx > 0  # can you actually close in x?
                    penalty = 0.0
                    if behind_now and not closing_x:
                        # Can't catch from behind in x with this action — soft penalty
                        penalty += behind_bias
                    # Also discourage overshooting ahead of the puck back at t*
                    agent_x_t = agent_x + agent_vx * t_star
                    puck_back_x_t = (puck_x - self.paddle_margin) + puck_vx * t_star
                    if agent_x_t > puck_back_x_t:
                        penalty += behind_bias * 0.5
                    score = dist2 + penalty
                else:
                    score = dist2

                if score < best_score:
                    best_score = score
                    best = (agent_x_cmd, agent_y_cmd)

        return best
    def down_puck_action(self, agent_x, agent_y, puck_x, puck_y) -> Tuple[int, int]:
        y_diff = puck_y - agent_y
        abs_y_diff = np.abs(y_diff)

        if abs_y_diff > self.paddle_margin + self.unit_speed_px:
            y = int(y_diff/abs_y_diff)
            if ((y == 1 and agent_y >= self.y_max - self.unit_speed_px) or
                    (y == -1 and agent_y <= self.y_min + self.unit_speed_px)):
                y = 0
            if agent_x <= self.x_min + self.unit_speed_px:
                return 0, y
            return -1, y
        if abs_y_diff == self.paddle_margin + self.unit_speed_px:
            if agent_x <= self.x_min + self.unit_speed_px:
                return 0, 0
            return -1, 0

        x_diff = puck_x - agent_x
        if abs_y_diff > self.unit_speed_px:
            if x_diff >= 0:
                if agent_x <= self.x_max - self.unit_speed_px:
                    return 1, int(y_diff / abs_y_diff)
                return 0, int(y_diff / abs_y_diff)
            if x_diff < -self.paddle_margin and agent_x >= self.x_min + self.unit_speed_px:
                return -1, -int(y_diff/abs_y_diff)
            return 0, -int(y_diff/abs_y_diff)
        diff_max = self.y_max - agent_y
        diff_min = agent_y - self.y_min
        if diff_max >= diff_min:
            y = 1
        else:
            y = -1
        if x_diff <= self.paddle_margin or agent_x <= self.x_min + self.unit_speed_px:
            return 0, y
        return -1, y
