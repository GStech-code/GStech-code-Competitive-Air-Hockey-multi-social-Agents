from typing import Dict, Tuple
from enum import IntEnum
import math
from .objective import Objective

class FastShot(Objective):
    """
    O1: Fast Shot (Direct)
    - Simple teammate avoidance, shallow wall bias, half-line guard.
    - ≤ 2 discrete commands per tick.
    """

    def __init__(self, agent_id, teammate_ids, commands, rules: Dict, **params):
        super().__init__(agent_id, commands, teammate_ids, rules, **params)

        # --- Environment ---
        width = rules.get("width", 800)
        height = rules.get("height", 600)
        self.unit = rules.get("unit_speed_px", 4.0)
        self.paddle_radius = rules.get("paddle_radius", 20)
        self.paddle_distance = self.paddle_radius * 2 + self.unit
        self.puck_radius = rules.get("puck_radius", 12)
        self.puck_distance = self.paddle_radius + self.puck_radius
        half_goal_gap = rules.get("goal_gap", 200) * 0.5

        # --- Single parameter: strike lane ---
        goal_gap_offest = params.get("goal_gap_offest", 0.6)
        wall_pad_add = params.get("wall_padding_add", 2)

        # --- Strike Y targets (inside the goal mouth) ---
        self.lane_center = height * 0.5
        self.goal_min = self.lane_center - half_goal_gap
        self.goal_max = self.lane_center + half_goal_gap
        half_goal_gap_offset = half_goal_gap * goal_gap_offest


        # --- Contact windows & pads ---
        self._hit_d = self.paddle_radius + self.puck_radius + self.unit
        self._hit_d2 = self._hit_d + self.unit
        self.double_unit = 2 * self.unit
        self._wall_pad = self.puck_radius + wall_pad_add
        self.up_wall_pad = height - self._wall_pad

        # --- Geometry constants ---
        self._half_line = width * 0.5
        self.quarter_line = width * 0.25
        self.x_goal = width - self.puck_radius
        self._fence_x = self._half_line - self.unit  # “safe” x we aim to stay ≤

        self.assume_puck_hit = False
        puck_max_speed = rules.get("puck_max_speed", 6)
        self.min_required_p_speed = min(self.unit, puck_max_speed / 3)
        self.min_sufficient_p_speed = min(self.unit, puck_max_speed / 2)

        self.last_pvx, self.last_pvy = 0, 0
        self.change_tol = math.cos(0.1)

    # ----------------- Helpers -----------------

    @staticmethod
    def _sign(v: float) -> int:
        return 0 if v == 0 else (1 if v > 0 else -1)

    def puck_dir_changed(self, pvx, pvy):
        # If both were or are static → no direction change
        if (
                (abs(pvx) < 1e-3 and abs(pvy) < 1e-3) and
                (abs(self.last_pvx) < 1e-3 and abs(self.last_pvy) < 1e-3)
        ):
            return False

        # If one vector is static and the other is not, direction has changed
        if ((abs(pvx) < 1e-3 and abs(pvy) < 1e-3) or
                (abs(self.last_pvx) < 1e-3 and abs(self.last_pvy) < 1e-3)):
            return True

        dot = (pvx * self.last_pvx + pvy * self.last_pvy) / (
                ((pvx ** 2 + pvy ** 2) * (self.last_pvx ** 2 + self.last_pvy ** 2)) ** 0.5 + 1e-9)
        return dot < self.change_tol


    def _teammate_y_nudge(self, ws: Dict, exp_x: float, exp_y: float) -> int:
        """±1 if teammate overlaps our small same-lane box; else 0."""
        if not self.teammate_ids:
            return 0
        x_min, x_max = exp_x - self.paddle_radius, exp_x + self.paddle_radius
        for tid in self.teammate_ids:
            tx, ty = ws["agent_x"][tid], ws["agent_y"][tid]
            if x_min <= tx <= x_max and abs(ty - exp_y) <= self.paddle_distance:
                return -1 if ty > exp_y else 1
        return 0

    def _guard_halfline(self, exp_x: float) -> int:
        """
        Only clamp forward motion (+x) so we never cross the half line.
        """
        if exp_x >= self._half_line:
            return 0
        next_x = exp_x + self.unit
        return 0 if next_x > self._half_line else 1

    def _guard_halfline_twice(self, exp_x: float) -> Tuple[int, int]:
        """
        Only clamp forward motion (+x) when we won't cross the half line.
        Snap back if over the line.
        """
        if exp_x >= self._half_line:
            return -1, 0
        exp_x += self.unit
        if exp_x >= self._half_line:
            return 0, 0
        exp_x += self.unit
        if exp_x >= self._half_line:
            return 1, 0
        return 1, 1

    def _y_at_x(self, px: float, py: float, pvx: float, pvy: float, target_x: float) -> float:
        """Linear projection of puck y at a target x; if pvx≈0, return current y."""
        if pvx == 0:
            return py
        t = (target_x - px) / pvx
        return py + pvy * t

    def _goal_predicted(self, px: float, py: float, pvx: float, pvy: float) -> bool:
        """
        Rough: if puck is heading to the right and would reach the goal line
        with y inside the goal gap.
        """
        y_at_goal = self._y_at_x(px, py, pvx, pvy, self.x_goal)
        return self.goal_min <= y_at_goal <= self.goal_max

    # ----------------- Main tick -----------------

    def emergency_step(self, ws: Dict, **params):
        ay = ws["agent_y"][self.agent_id]
        py = ws["puck_y"]
        pvy = ws["puck_vy"]

        fut_py = py + pvy
        if fut_py <= ay - self.paddle_radius:
            self.commands.push((1, -1))
        elif fut_py >= ay + self.paddle_radius:
            self.commands.push((1, 1))
        else:
            self.commands.push((1, 0))

        self.intro_step(ws, **params)
    def intro_step(self, ws: Dict, **params):
        self.last_ws = ws
        self.assume_puck_hit = False
        self.last_pvx, self.last_pvy = ws["puck_vx"], ws["puck_vy"]
        self.continue_step()

    def new_ws_step(self, ws: Dict):
        self.last_ws = ws
        ax = ws["agent_x"][self.agent_id]
        px = ws["puck_x"]
        pvx = ws["puck_vx"]
        pvy = ws["puck_vy"]
        if (not self.commands.is_empty() and self.puck_dir_changed(pvx, pvy)) or (self.assume_puck_hit and (px <= ax)):
            self.assume_puck_hit = False
            self.commands.clear()
        self.last_pvx, self.last_pvy = pvx, pvy
        self.continue_step()

    def continue_step(self):
        """
        Behavior by regions (all with half-line guard):
          1) Puck behind us → sidestep then re-engage back (avoid self-hit).
          2) Enemy half, pvx>0 → likely shot: hold half-line, align Y (no crossing).
          3) Enemy half, pvx<=0 → coming back: align to intercept at half-line (no crossing).
          4) Otherwise (our half / near x): approach/hit logic with half-line guard.
        """
        # --- Unpack ---
        ws = self.last_ws
        ax = ws["agent_x"][self.agent_id]
        ay = ws["agent_y"][self.agent_id]
        px = ws["puck_x"]
        py = ws["puck_y"]
        pvx = ws["puck_vx"]
        pvy = ws["puck_vy"]

        # Expected after queued cmds
        adv_x, adv_y = self.commands.get_advance()
        exp_x = ax + adv_x * self.unit
        exp_y = ay + adv_y * self.unit

        # Assume hit, commands will change if no hit or actual miss
        if self.assume_puck_hit:
            ux = -1 if exp_x > self._fence_x else 0
            lane_y = self.lane_center
            uy = 0 if abs(lane_y - exp_y) <= self.unit else self._sign(lane_y - exp_y)
            yn = self._teammate_y_nudge(ws, exp_x, exp_y)
            if yn != 0:
                uy = yn
            self.commands.push((ux, uy))
            if ux == 0 and uy == 0:
                self.long_term_mode()
            return

        # Short horizon projection (when our next push lands)
        future_steps = self.commands.get_size() + 1
        fut_px = px + pvx * future_steps
        fut_py = py + pvy * future_steps

        # -------- 1) Puck behind us --------
        if fut_px < exp_x:
            dy = fut_py - exp_y
            # --- Step 1: basic back-hit safety ---
            if (exp_x - fut_px <= self._hit_d2) and (abs(dy) <= self._hit_d2):
                # Move slightly forward unless blocked by half-line
                ux1, ux2 = self._guard_halfline_twice(exp_x)
                uy = 1 if dy < 0 else -1
                self.commands.push_multiple([(ux1, uy), (ux2, uy)])
                return

            # --- Step 2: decide repositioning style ---
            # if puck is too deep (beyond quarter board), don’t chase; reposition to center-lane
            quarter_line = self.quarter_line
            if fut_px < quarter_line:
                # Stay upper field; align to chosen strike lane (don’t chase to bottom)
                lane_y = self.lane_center
                uy = 0 if abs(lane_y - exp_y) <= self.unit else self._sign(lane_y - exp_y)

                # mild retreat to prep for re-entry (never cross half)
                ux1, ux2 = self._guard_halfline_twice(exp_x - self.unit)
                self.commands.push_multiple([(ux1, uy), (ux2, uy)])
                return

            # --- Step 3: consider puck velocity direction ---
            # toward enemy → step forward to prep for rebound
            # toward our goal → retreat slightly to intercept
            if pvx > 0:
                ux1, ux2 = self._guard_halfline_twice(exp_x)
            else:
                ux1, ux2 = -1, -1  # step back defensively

            # --- Step 4: teammate avoidance ---
            uy = 0 if abs(dy) <= self.unit else self._sign(dy)
            y_nudge = self._teammate_y_nudge(ws, exp_x, exp_y)
            if y_nudge != 0:
                uy = y_nudge

            # --- Step 5: final push (2 ticks) ---
            self.commands.push_multiple([(ux1, uy), (ux2, uy)])
            return


        if px >= self._half_line:
            # Never cross half-line; if we drifted over, pull back once, else hold x.
            ux = -1 if exp_x > self._fence_x else 0
            # -------- 2) Enemy half & going to enemy goal (pvx>0) --------
            if pvx > 0:
                # If it's probably a goal already: hold line, align to rebound-friendly Y (center lane).
                # Else: still hold line; align toward our chosen strike lane to be ready for turnover.
                target_y = self.lane_center
                dy = target_y - exp_y
                uy = 0 if abs(dy) <= self.unit else self._sign(dy)
                # Small teammate/wall adjustments on the chosen uy
                yn = self._teammate_y_nudge(ws, exp_x, exp_y)
                if yn != 0:
                    uy = yn
                elif uy == 0:
                    if py <= self._wall_pad:
                        uy = +1
                    if py >= self.up_wall_pad:
                        uy = -1

            # -------- 3) Enemy half & coming back (pvx<=0): prep intercept at half-line --------
            else:
                y_half = self._y_at_x(px, py, pvx, pvy, self._half_line)
                dy = y_half - exp_y
                uy = 0 if abs(dy) <= self.unit else self._sign(dy)
                yn = self._teammate_y_nudge(ws, exp_x, exp_y)
                if yn != 0:
                    uy = yn

            self.commands.push_multiple([(ux, uy), (ux, uy)])
            return

        # -------- 4) puck in team half is moving fast enough forward, no need to interfere
        if (pvx >= self.min_sufficient_p_speed or
                (pvx >= self.min_required_p_speed and pvy >= self.min_required_p_speed)):
            dy = fut_py - exp_y
            if (fut_px - exp_x <= self._hit_d2) and (abs(dy) <= self._hit_d2):
                # Move slightly backwards if about to hit the puck
                uy = 1 if dy < 0 else -1
                ux = -1
            else:
                ux = -1 if exp_x > self._fence_x else 0
                lane_y = self.lane_center
                uy = 0 if abs(lane_y - exp_y) <= self.unit else self._sign(lane_y - exp_y)
                yn = self._teammate_y_nudge(ws, exp_x, exp_y)
                if yn != 0:
                    uy = yn
            self.commands.push_multiple([(ux, uy), (ux, uy)])
            if ux == 0 and uy == 0:
                self.long_term_mode()
            return

        # -------- 5) Our half / near-x: attempt hit, else approach/align --------
        # Contact attempt
        if abs(fut_px - exp_x) <= self._hit_d and abs(fut_py - exp_y) <= self._hit_d:
            uy = 0 if abs(fut_py - exp_y) <= self.unit else self._sign(fut_py - exp_y)
            # teammate + wall simple shaping
            yn = self._teammate_y_nudge(ws, exp_x, exp_y)
            if yn != 0:
                uy = yn
            else:
                self.assume_puck_hit = True

            ux1, ux2 = self._guard_halfline_twice(exp_x)
            self.commands.push_multiple([(ux1, uy), (ux2, uy)])
            return

        # Pre-hit shaping (close x gap + align Y to lane) with half-line guard
        dx = fut_px - exp_x
        dy = fut_py - exp_y
        if abs(dy) <= self.double_unit and dx > self.double_unit:
            lane_y = self.lane_center
            dy = lane_y - exp_y
        uy = 0 if abs(dy) <= self.unit else self._sign(dy)

        if dx > 0:
            ux1, ux2 = self._guard_halfline_twice(exp_x)  # +1 or 0 if crossing half
        else:
            ux1, ux2 = -1, -1

        # Teammate safety override (only if overlapping our small lane box)
        yn = self._teammate_y_nudge(ws, exp_x, exp_y)
        if yn != 0:
            uy = yn

        self.commands.push_multiple([(ux1, uy), (ux2, uy)])
