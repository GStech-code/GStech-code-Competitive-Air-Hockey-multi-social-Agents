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

    def __init__(self, agent_id, teammate_ids, rules: Dict, **params):
        super().__init__(agent_id, teammate_ids, rules, **params)

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
        wall_pad_add = params.get("wall_padding_add", 2)

        # --- Strike Y targets (inside the goal mouth) ---
        self.lane_center = height * 0.5
        self.goal_min = self.lane_center - half_goal_gap
        self.goal_max = self.lane_center + half_goal_gap

        # --- Contact windows & pads ---
        self._hit_d = self.paddle_radius + self.puck_radius + self.unit
        self._hit_d2 = self._hit_d + self.unit
        self.double_unit = 2 * self.unit
        self._wall_pad = self.puck_radius + wall_pad_add
        self.up_wall_pad = height - self._wall_pad

        # --- Geometry constants ---
        self._half_line = width * 0.5
        self.enforcement_line = params.get('enforcement_line', width * 0.375)
        self.quarter_line = width * 0.25
        self.x_goal = width - self.puck_radius
        self._fence_x = self._half_line - self.unit  # “safe” x we aim to stay ≤

        puck_max_speed = rules.get("puck_max_speed", 6)
        self.min_required_p_speed = min(self.unit, puck_max_speed / 3)
        self.min_sufficient_p_speed = min(self.unit, puck_max_speed / 2)


    # ----------------- Helpers -----------------

    @staticmethod
    def _sign(v: float) -> int:
        return 0 if v == 0 else (1 if v > 0 else -1)

    def down_y_advance_safe(self, dy, ady):
        if ady > self._hit_d2:
            return self._sign(dy)
        elif ady == self._hit_d2:
            return 0
        return self._sign(-dy)

    def _teammate_y_nudge(self, ws: Dict, ax: float, ay: float) -> int:
        """±1 if teammate overlaps our small same-lane box; else 0."""
        if not self.teammate_ids:
            return 0
        x_min, x_max = ax - self.paddle_radius, ax + self.paddle_radius
        for tid in self.teammate_ids:
            tx, ty = ws["agent_x"][tid], ws["agent_y"][tid]
            if x_min <= tx <= x_max and abs(ty - ay) <= self.paddle_distance:
                return -1 if ty > ay else 1
        return 0

    def _to_enforcement(self, ax: float) -> int:
        if ax >= self.enforcement_line:
            return -1
        next_x = ax + self.unit
        return 0 if next_x > self.enforcement_line else 1

    def _guard_halfline(self, ax: float) -> int:
        """
        Only clamp forward motion (+x) so we never cross the half line.
        """
        if ax >= self._half_line:
            return -1
        next_x = ax + self.unit
        return 0 if next_x > self._half_line else 1

    def _y_at_x(self, px: float, py: float, pvx: float, pvy: float, target_x: float) -> float:
        """Linear projection of puck y at a target x; if pvx≈0, return current y."""
        if pvx == 0:
            return py
        t = (target_x - px) / pvx
        return py + pvy * t

    # ----------------- Main tick -----------------

    def step(self, ws: Dict) -> Tuple[int, int]:
        """
        Behavior by regions (all with half-line guard):
          1) Puck behind us → sidestep then re-engage back (avoid self-hit).
          2) Enemy half, pvx>0 → likely shot: hold half-line, align Y (no crossing).
          3) Enemy half, pvx<=0 → coming back: align to intercept at half-line (no crossing).
          4) Otherwise (our half / near x): approach/hit logic with half-line guard.
        """
        # --- Unpack ---
        ax = ws["agent_x"][self.agent_id]
        ay = ws["agent_y"][self.agent_id]
        px = ws["puck_x"]
        py = ws["puck_y"]
        pvx = ws["puck_vx"]
        pvy = ws["puck_vy"]

        # Short horizon projection (when our next push lands)
        fut_px = px + pvx
        fut_py = py + pvy

        # -------- 1) Puck behind us --------
        if fut_px < ax:
            dy = fut_py - ay
            ady = abs(dy)
            # --- Step 1: basic back-hit safety ---
            if (ax - fut_px <= self._hit_d2) and (abs(dy) <= self._hit_d2):
                # Move slightly forward unless blocked by half-line
                ux = self._guard_halfline(ax)
                uy = 1 if dy < 0 else -1
                return ux, uy

            # --- Step 2: decide repositioning style ---
            # if puck is too deep (beyond quarter board), don’t chase; reposition to center-lane
            quarter_line = self.quarter_line
            if fut_px < quarter_line and abs(pvx) > self.min_sufficient_p_speed:
                # Stay upper field; align to chosen strike lane (don’t chase to bottom)
                lane_y = self.lane_center
                uy = 0 if abs(lane_y - ay) <= self.unit else self._sign(lane_y - ay)

                # mild retreat to prep for re-entry (never cross half)
                ux = self._to_enforcement(ax)
                return ux, uy

            # --- Step 3: consider puck velocity direction ---
            # toward enemy → step forward to prep for rebound
            # toward our goal → retreat slightly to intercept
            if pvx > self.min_required_p_speed:
                ux= self._guard_halfline(ax)
            else:
                ux = -1 # step back defensively

            # --- Step 4: teammate avoidance ---
            uy = self.down_y_advance_safe(dy, ady)
            y_nudge = self._teammate_y_nudge(ws, ax, ay)
            if y_nudge != 0:
                uy = y_nudge

            # --- Step 5: final push (2 ticks) ---
            return ux, uy


        if px >= self._half_line:
            # Never cross half-line; if we drifted over, pull back once, else hold x.
            ux = -1 if ax > self._fence_x else 0
            # -------- 2) Enemy half & going to enemy goal (pvx>0) --------
            if pvx > 0:
                # If it's probably a goal already: hold line, align to rebound-friendly Y (center lane).
                # Else: still hold line; align toward our chosen strike lane to be ready for turnover.
                target_y = self.lane_center
                dy = target_y - ay
                uy = 0 if abs(dy) <= self.unit else self._sign(dy)
                # Small teammate/wall adjustments on the chosen uy
                yn = self._teammate_y_nudge(ws, ax, ay)
                if yn != 0:
                    uy = yn
                elif uy == 0:
                    if py <= self._wall_pad:
                        uy = 1
                    if py >= self.up_wall_pad:
                        uy = -1

            # -------- 3) Enemy half & coming back (pvx<=0): prep intercept at half-line --------
            else:
                y_half = self._y_at_x(px, py, pvx, pvy, self._half_line)
                dy = y_half - ay
                uy = 0 if abs(dy) <= self.unit else self._sign(dy)
                yn = self._teammate_y_nudge(ws, ax, ay)
                if yn != 0:
                    uy = yn

            return ux, uy

        # -------- 4) puck in team half is moving fast enough forward, no need to interfere
        if (pvx >= self.min_sufficient_p_speed or
                (pvx >= self.min_required_p_speed and pvy >= self.min_required_p_speed)):
            dy = fut_py - ay
            if (fut_px - ax <= self._hit_d2) and (abs(dy) <= self._hit_d2):
                # Move slightly backwards if about to hit the puck
                uy = 1 if dy < 0 else -1
                ux = -1
            else:
                ux = -1 if ax > self._fence_x else 0
                lane_y = self.lane_center
                uy = 0 if abs(lane_y - ay) <= self.unit else self._sign(lane_y - ay)
                yn = self._teammate_y_nudge(ws, ax, ay)
                if yn != 0:
                    uy = yn
            return ux, uy

        # -------- 5) Our half / near-x: attempt hit, else approach/align --------
        # Contact attempt
        if abs(fut_px - ax) <= self._hit_d and abs(fut_py - ay) <= self._hit_d:
            uy = 0 if abs(fut_py - ay) <= self.unit else self._sign(fut_py - ay)
            # teammate + wall simple shaping
            yn = self._teammate_y_nudge(ws, ax, ay)
            if yn != 0:
                uy = yn

            ux = self._guard_halfline(ax)
            return ux, uy

        # Pre-hit shaping (close x gap + align Y to lane) with half-line guard
        dx = fut_px - ax
        dy = fut_py - ay
        ady = abs(dy)
        if abs(dy) <= self.double_unit and dx > self.double_unit:
            lane_y = self.lane_center
            dy = lane_y - ay


        if dx > self.unit:
            ux = self._guard_halfline(ax)  # +1 or 0 if crossing half
            uy = 0 if abs(dy) <= self.unit else self._sign(dy)
        else:
            ux = -1
            uy = self.down_y_advance_safe(dy, ady)

        # Teammate safety override (only if overlapping our small lane box)
        yn = self._teammate_y_nudge(ws, ax, ay)
        if yn != 0:
            uy = yn

        return ux, uy
