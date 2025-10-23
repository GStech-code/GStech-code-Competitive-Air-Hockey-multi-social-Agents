# intercept.py
from typing import Dict, Tuple
import math
from .objective import Objective

class Intercept(Objective):
    """
    D2: Intercept (Defensive, parameter-light)
    - If puck outbound (pvx > 0): avoid contact; posture back toward a defensive home-X.
    - If inbound (pvx <= 0): sprint to earliest safe contact (short horizon) WITHOUT crossing half-line.
    - Assume-intercept is used, but auto-cancelled if direction unchanged yet contact won't happen.
    - ≤ 2 cmds per tick. Uses FastShot-style helpers/guards.
    """

    def __init__(self, agent_id, teammate_ids, rules: Dict, **params):
        super().__init__(agent_id, teammate_ids, rules, **params)

        # --- Environment / geometry (mirror FastShot naming) ---
        width = rules.get("width", 800)
        height = rules.get("height", 600)
        self.unit = rules.get("unit_speed_px", 4.0)
        self.paddle_radius = rules.get("paddle_radius", 20)
        self.puck_radius = rules.get("puck_radius", 12)

        # Contact windows & pads
        self._hit_d = self.paddle_radius + self.puck_radius + self.unit
        self._hit_d2 = self._hit_d + self.unit
        self._up_wall_puck = height - self.puck_radius
        self._low_wall_puck = self.puck_radius

        # Board references
        self._half_line = width * 0.5
        self.y_center = height * 0.5

        # Defensive posture (only in safe/outbound contexts)
        self.x_home = params.get("x_home", width * 0.25)
    # ----------------- Helpers (mirroring FastShot) -----------------

    @staticmethod
    def _sign(v: float) -> int:
        return 0 if v == 0 else (1 if v > 0 else -1)

    def down_y_advance_safe(self, dy, ady):
        if ady > self._hit_d2:
            return self._sign(dy)
        elif ady == self._hit_d2:
            return 0
        return self._sign(-dy)

    def _teammate_y_nudge(self, ws: Dict, exp_x: float, exp_y: float) -> int:
        """±1 if teammate overlaps our small same-lane box; else 0."""
        if not self.teammate_ids:
            return 0
        x_min, x_max = exp_x - self.paddle_radius, exp_x + self.paddle_radius
        for tid in self.teammate_ids:
            tx, ty = ws["agent_x"][tid], ws["agent_y"][tid]
            if x_min <= tx <= x_max and abs(ty - exp_y) <= (self.paddle_radius * 2 + self.unit):
                return -1 if ty > exp_y else 1
        return 0

    def _guard_halfline(self, exp_x: float) -> int:
        """Clamp forward (+x) so we never cross the half line."""
        if exp_x >= self._half_line:
            return -1
        nxt = exp_x + self.unit
        return 0 if nxt > self._half_line else 1

    def _project_y_with_wall_once(self, py: float, pvy: float) -> float:
        """
        One-bounce vertical projection for short horizon:
        - Move to py + pvy*steps.
        - If passes a wall once, reflect around that wall and stop.
        """
        raw = py + pvy
        # If within pads, no bounce
        if self._low_wall_puck <= raw <= self._up_wall_puck:
            return raw
        # Crossed top?
        if raw > self._up_wall_puck and pvy > 0:
            over = raw - self._up_wall_puck
            return self._up_wall_puck - over  # reflect
        # Crossed bottom?
        if raw < self._low_wall_puck and pvy < 0:
            over = self._low_wall_puck - raw
            return self._low_wall_puck + over  # reflect
        return raw

    # ----------------- Main -----------------
    def step(self, ws: Dict) -> Tuple[int, int]:
        # --- Unpack ---
        agents_x = ws["agent_x"]
        agents_y = ws["agent_y"]
        ax = agents_x[self.agent_id]
        ay = agents_y[self.agent_id]
        px = ws["puck_x"]
        py = ws["puck_y"]
        pvx = ws["puck_vx"]
        pvy = ws["puck_vy"]

        # Short-horizon projection (when our next push lands) + single wall reflection
        fut_px = px + pvx
        fut_py = self._project_y_with_wall_once(py, pvy)

        # ===================== A) Outbound puck (avoid contact) =====================
        dx = fut_px - ax
        dy = fut_py - ay
        ady = abs(dy)

        teammate_close = False
        for id in self.teammate_ids:
            tx, ty = agents_x[id], agents_y[id]
            if fut_px - tx <= self._hit_d2 and abs(fut_py - ty) <= self._hit_d2:
                teammate_close = True
                break

        if pvx > 0 or teammate_close:

            # If we'd collide soon, sidestep away from puck Y and mildly retreat X
            if (abs(dx) <= self._hit_d2) and (ady <= self._hit_d2):
                ux = -1 if ax > self.paddle_radius else 0
                # Your requested dodge rule: move away from puck Y
                uy = 1 if dy < 0 else -1
                return ux, uy

            # Otherwise: posture toward defensive home when puck not close to our half
            # "Not close to our half": future puck x beyond half-line margin.
            if fut_px > self._half_line:
                dxh = self.x_home - ax
                ux = 0 if abs(dxh) <= self.unit else self._sign(dxh)
            else:
                # drift to x_home
                ux = -1 if ax > self.x_home else 0
            dyh = self.y_center - ay
            uy = 0 if abs(dyh) <= self.unit else self._sign(dyh)
            yn = self._teammate_y_nudge(ws, ax, ay)
            if yn != 0:
                uy = yn
            return ux, uy

        # ===================== B) Inbound puck (intercept earliest safe) =====================


        # If behind us: basic back-hit safety first (don’t let it clip from behind)
        r_dx = ax - fut_px
        if fut_px < ax and (r_dx <= self._hit_d2) and (ady <= self._hit_d2):
            if r_dx <= self._hit_d and ady >= self._hit_d:
                return -1, 0
            ux = self._guard_halfline(ax)
            # Move away from incoming puck Y (mirror your outbound dodge; protects from self-hit)
            uy = 1 if dy < 0 else -1
            return ux, uy

        # Choose minimal-move burst toward projected contact, with half-line guard
        ux = 0 if abs(dx) <= self.unit else self._sign(dx)
        if ux == 1:
            ux = self._guard_halfline(ax)  # forbid crossing half
            uy = 0 if abs(dy) <= self.unit else self._sign(dy)
        elif ux == -1:
            uy = self.down_y_advance_safe(dy, ady)
        else:
            uy = 0 if abs(dy) <= self.unit else self._sign(dy)

        # Teammate lane overlap → nudge Y
        yn = self._teammate_y_nudge(ws, ax, ay)
        if yn != 0:
            uy = yn
        return ux, uy

