from typing import Dict, List
from math import inf, exp, hypot

DIVIDE_BY_THREE = 1.0 / 3.0

class UCTHeuristic:
    """
    Positive favors Team A, negative favors Team B.

    Final score = (team_a_score - team_a_score_before) - (team_b_score - team_b_score_before) + H,
    where H ∈ [-1, 1] is the heuristic composed of three parts, each in [-1/3, +1/3]:
      1) Puck field position (closer to enemy side is better)
      2) Paddles-behind-puck (A wants x_agent <= x_puck, B wants x_agent >= x_puck)
      3) Near-term goal chance vs. being blocked (with simple bounce + reachability)
    """

    def __init__(self, team_a_agent_ids: List[int], team_b_agent_ids: List[int], **params):
        self.A = team_a_agent_ids
        self.len_a = len(team_a_agent_ids)
        self.B = team_b_agent_ids
        self.len_b = len(team_b_agent_ids)
        self.W = float(params.get('width', 800))
        self.H = float(params.get('width', 600))
        goal_gap = float(params.get('goal_gap', 240))

        self.goal_cy = self.H * 0.5  # center fixed at mid-height
        self.goal_half = 0.5 * goal_gap

        self.us = float(params.get('unit_speed_px', 4))
        self.Rp = float(params.get('paddle_radius', 20))
        self.Rk = float(params.get('puck_radius', 12))
        self.puck_distance = self.Rp + self.Rk
        self.H_protected = self.H - (2 * self.Rk)
        self.H_period = self.H_protected * 2

        self.half_line = self.W * 0.5
        self.half_line_scale = self.half_line - self.Rk

        # precompute useful bounds
        self.min_x = 0.0
        self.max_x = self.W

    # ---------- small utilities ----------

    @staticmethod
    def _clip(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def _triangle_bounce_y(self, y0: float, vy: float, t: float) -> float:
        """ Horizontal-wall reflections on [0,H] using triangle-wave folding. O(1), no loops. """
        R = self.Rk
        if abs(vy) < 1e-12 or t <= 0.0:
            return y0
        period = self.H_period
        y_rel0 = y0 - R
        m = (y_rel0 + vy * t) % period
        # Python's % already returns non-negative
        y_fold = m if m <= self.H_protected else (period - m)
        return y_fold + R

    @staticmethod
    def _time_to_x(x_target, x0, vx):
        if abs(vx) < 1e-12:
            return inf
        t = (x_target - x0) / vx
        return t if t >= 0.0 else inf

    # ---------- Part 1: puck field position ----------

    def _h_field_position(self, px):
        # Normalize puck x around midline: +1 means fully near B goal (good for A), -1 near A goal.
        half = self.half_line
        # keep a small margin for radius so extremes map cleanly to ±1
        scale = self.half_line_scale
        s = self._clip((px - half) / scale, -1.0, 1.0)
        return DIVIDE_BY_THREE * s

    # ---------- Part 2: paddles behind the puck ----------

    def _h_behind(self, px, agent_x):
        # A is "behind" if x <= px - Rp; B is "behind" if x >= px + Rp
        if self.A:
            a_behind = sum(1 for i in self.A if agent_x[i] <= px - self.Rp) / self.len_a
        else:
            a_behind = 0.0
        if self.B:
            b_behind = sum(1 for i in self.B if agent_x[i] >= px + self.Rp) / self.len_b
        else:
            b_behind = 0.0
        # favor A when A-behind fraction > B-behind fraction
        return DIVIDE_BY_THREE * self._clip(a_behind - b_behind, -1.0, 1.0)

    # ---------- Part 3: goal likelihood vs. block ----------

    def _can_block_right(self, px, py, vx, vy, agent_x, agent_y):
        """
        Can any B defender plausibly block puck traveling to the RIGHT goal?
        We check each defender whose x is in [px, W] and compare:
          |y_def - y_puck_at_x_def| <= Rp + Rk + us * t_to_def  (y reach allowance)
        """
        if vx <= 0.0:
            return False
        for j in self.B:
            xj = agent_x[j]
            if xj < px:
                continue
            t = self._time_to_x(xj, px, vx)
            if t == inf:
                continue
            y_at = self._triangle_bounce_y(py, vy, t)
            # defender vertical reach during time t
            reach = self.puck_distance + self.us * t
            if abs(agent_y[j] - y_at) <= reach:
                return True
        return False

    def _can_block_left(self, px, py, vx, vy, agent_x, agent_y):
        """ Mirror of _can_block_right for LEFT goal. """
        if vx >= 0.0:
            return False
        for j in self.A:
            xj = agent_x[j]
            if xj > px:
                continue
            t = self._time_to_x(xj, px, vx)
            if t == inf:
                continue
            y_at = self._triangle_bounce_y(py, vy, t)
            reach = self.puck_distance + self.us * t
            if abs(agent_y[j] - y_at) <= reach:
                return True
        return False

    def _goal_chance_diff(self, px, py, vx, vy, agent_x, agent_y):
        """
        Returns value in [-1/3, +1/3]:
          + when A likely to score soon (right goal) and not blocked,
          - when B likely to score soon (left goal) and not blocked.
        Uses exponential time discount; crude 0/1 window test for goal lane.
        """
        if abs(vx) <= 0.01:
            return 0  # Puck is not moving towards any goal fast enoguh
        vmag = hypot(vx, vy)

        # A scoring (RIGHT)
        score = 0
        if vx > 0.0:
            tR = self._time_to_x(self.max_x, px, vx)
            if tR is not inf:
                yR = self._triangle_bounce_y(py, vy, tR)
                in_lane = abs(yR - self.goal_cy) <= self.goal_half
                blocked = self._can_block_right(px, py, vx, vy, agent_x, agent_y)
                if in_lane and not blocked:
                    score = exp(-(vmag * tR) / self.W)

        # B scoring (LEFT)
        else:  # vx < 0.0
            tL = self._time_to_x(self.min_x, px, vx)
            if tL is not inf:
                yL = self._triangle_bounce_y(py, vy, tL)
                in_lane = abs(yL - self.goal_cy) <= self.goal_half
                blocked = self._can_block_left(px, py, vx, vy, agent_x, agent_y)
                if in_lane and not blocked:
                    score = -exp(-(vmag * tL) / self.W)

        return DIVIDE_BY_THREE * self._clip(score, -1.0, 1.0)

    # ---------- Public API ----------

    def evaluate(self, team_a_score_before: int, team_b_score_before: int, world_state: Dict):
        """
        world_state keys: team_a_score, team_b_score, puck_x, puck_y, puck_vx, puck_vy, agent_x, agent_y
        agent_x/agent_y are indexable by global agent ids (matching self.A / self.B).
        """
        a_now = world_state["team_a_score"]
        b_now = world_state["team_b_score"]

        # objective delta
        delta = (a_now - team_a_score_before) - (b_now - team_b_score_before)

        # unpack puck & agents
        px = float(world_state["puck_x"])
        py = float(world_state["puck_y"])
        vx = float(world_state["puck_vx"])
        vy = float(world_state["puck_vy"])
        agent_x = world_state["agent_x"]
        agent_y = world_state["agent_y"]

        # heuristic parts
        h1 = self._h_field_position(px)
        h2 = self._h_behind(px, agent_x)
        h3 = self._goal_chance_diff(px, py, vx, vy, agent_x, agent_y)

        H = self._clip(h1 + h2 + h3, -1.0, 1.0)

        return delta + H
