# pass_shot.py
from typing import Dict, List, Tuple
from .objective import Objective
import math

class PassShot(Objective):
    """
    M1: Pass
    Purpose: Feed the puck toward a teammate when safe and meaningful.
    - No params, only internal factors.
    - Avoids overlapping teammates using stronger multi-nudge logic.
    - Positions near the center band, drifts dynamically when idle.
    """

    def __init__(self, agent_id, teammate_ids, rules: Dict, **params):
        super().__init__(agent_id, teammate_ids, rules, **params)
        self.width = rules.get("width", 800)
        self.height = rules.get("height", 600)
        self.unit = rules.get("unit_speed_px", 4.0)
        self.paddle_r = rules.get("paddle_radius", 20)
        self.puck_r = rules.get("puck_radius", 12)
        self.teammate_distance = (self.paddle_r + self.unit) * 2
        self.y_center = self.height / 2.0
        self.half_line = self.width * 0.5
        self.hold_line = self.half_line - self.paddle_r - self.unit
        self.half_unit = self.unit * 0.5
        self.x_home = self.width * 0.45  # central stance, slightly defensive

        # short memory
        self._hit_d2 = self.paddle_r + self.puck_r + (self.unit * 2)
        self._low_wall = self.puck_r
        self._up_wall = self.height - self.puck_r

    # ----------------- Helpers -----------------

    def _sign(self, v: float) -> int:
        return 0 if abs(v) < 1e-6 else (1 if v > 0 else -1)

    def down_y_advance_safe(self, dy, ady):
        if ady > self._hit_d2:
            return self._sign(dy)
        elif ady == self._hit_d2:
            return 0
        return self._sign(-dy)

    def _multi_teammate_y_nudge(self, ax: float, ay: float) -> int:
        """
        Stronger multi-nudge:
        Applies repulsion from all teammates (distance-weighted),
        keeping this paddle centered between them when possible.
        """
        if not self.teammate_ids:
            return 0
        total_force = 0.0
        for tid in self.teammate_ids:
            tx = self.a_x[tid]
            ty = self.a_y[tid]
            dx = abs(tx - ax)
            dy = ty - ay
            # influence only if close enough in X
            if dx < self.teammate_distance:
                weight = 1.0 / (1.0 + dx)
                total_force += -math.copysign(weight, dy)
        if abs(total_force) < 0.2:
            return 0
        return 1 if total_force > 0 else -1

    def _project_y_with_wall_once(self, py: float, pvy: float) -> float:
        """Simple single-bounce projection, identical to others."""
        raw = py + pvy
        if self._low_wall <= raw <= self._up_wall:
            return raw
        if raw > self._up_wall:
            over = raw - self._up_wall
            return self._up_wall - over
        if raw < self._low_wall:
            over = self._low_wall - raw
            return self._low_wall + over
        return raw

    # ----------------- Main -----------------

    def step(self, ws: Dict) -> Tuple[int, int]:
        """
        Maintain central supportive stance; attempt passes when conditions align.
        """
        # --- Unpack ---
        a_x = ws["agent_x"]
        self.a_x = a_x
        a_y = ws["agent_y"]
        self.a_y = a_y
        ax = a_x[self.agent_id]
        ay = a_y[self.agent_id]
        px = ws["puck_x"]
        py = ws["puck_y"]
        pvx = ws["puck_vx"]
        pvy = ws["puck_vy"]

        # Short-horizon projection (when our next push lands) + single wall reflection
        fut_px = px + pvx
        fut_py = self._project_y_with_wall_once(py, pvy)

        dx = fut_px - ax
        dy = fut_py - ay
        pvx_out = (pvx >= 0)  # outbound (to the right)
        pv_fast = (pvx > self.half_unit and pvy >= self.half_unit)

        # Closest receiver (self vs teammates)

        # ===== 1) Puck beyond hold line =====
        if fut_px > self.hold_line:
            if pvx_out:
                # (A) Outbound & already past hold: reposition/hold; don't chase
                ux = -1 if ax > self.x_home else 0
                uy = 0 if abs(self.y_center - ay) <= self.unit else self._sign(self.y_center - ay)
                yn = self._multi_teammate_y_nudge(ax, ay)
                if yn:
                    uy = yn
                return ux, uy
            else:
                # (B) Inbound but already past hold: prep to defend/redirect (no half-line crossing)
                ux = 0  # never +x here
                uy = 0 if abs(fut_py - ay) <= self.unit else self._sign(dy)
                yn = self._multi_teammate_y_nudge(ax, ay)
                if yn:
                    uy = yn
                return ux, uy

        # ===== 2) Puck behind us in x (left of paddle) =====
        if fut_px < ax:
            # Distances in our frame
            adx = ax - fut_px  # how far behind in X
            ady = abs(dy)

            # Too close from behind → immediate avoidance: step AWAY in Y, never advance in X
            if (adx <= self._hit_d2) and (ady <= self._hit_d2):
                uy = 1 if dy < 0 else -1  # move away from puck's Y (no teammate override here)
                ux = -1 if ax > self.paddle_r else 0  # mild retreat; NEVER +x in this subcase
                return ux, uy

            # Otherwise: do NOT back-chase; central hold with teammate respect
            ux = -1 if ax > self.x_home else 0
            uy = self.down_y_advance_safe(dy, ady)
            yn = self._multi_teammate_y_nudge(ax, ay)
            if yn: uy = yn
            return ux, uy

        # ===== 3) Fast inbound (both components meaningful) =====
        if pv_fast:
            # Stay supportive: no +x chase; gently align Y; stronger teammate yield
            ux = -1 if ax > self.x_home else 0  # at most retreat, never advance here
            uy = self.down_y_advance_safe(dy, abs(dy))
            yn = self._multi_teammate_y_nudge(ax, ay)
            if yn:
                uy = yn
            return ux, uy

        # ===== 4) Special down-heading cases near us vs teammate =====
        # (a) Puck above us (smaller y), heading down, and closest to US → prepare to feed/redirect
        d_self = (dx * dx + dy * dy) ** 0.5
        best_tm = float("inf")
        for tid in self.teammate_ids:
            tx, ty = a_x[tid], a_y[tid]
            d = ((fut_px - tx) ** 2 + (fut_py - ty) ** 2) ** 0.5
            if d < best_tm:
                best_tm = d
        teammate_closest = d_self > best_tm
        #Puck heading down but closer to a TEAMMATE → yield lane, support centrally
        if teammate_closest:
            ux = -1 if ax > self.x_home else 0
            uy = self._sign(self.y_center - ay) if abs(self.y_center - ay) > self.unit else 0
            yn = self._multi_teammate_y_nudge(ax, ay)
            if yn:
                uy = yn
            return ux, uy

        # ===== 5) Default pass =====
        # Try to PASS to nearest teammate above us; else kill downward motion; else light converge

        # 1) Nearest ABOVE teammate (ty < exp_y)
        recv = None
        best = 1e18
        for tid in self.teammate_ids:
            tx, ty = a_x[tid], a_y[tid]
            if tx > ax:  # "above" == ahead toward enemy goal
                d2 = (tx - ax) ** 2 + (ty - ay) ** 2
                if d2 < best:
                    best, recv = d2, (tx, ty)

        if recv:
            tx, ty = recv

            # Vector paddle→puck now (predicted), and puck→teammate (desired travel dir)
            dx, dy = fut_px - ax, fut_py - ay
            tdx, tdy = tx - fut_px, ty - fut_py

            close_enough_to_strike = (abs(dx) <= self.unit and abs(dy) <= self.unit)

            if close_enough_to_strike:
                # "Guess hit": nudge puck *toward teammate* with one command.
                # Respect half-line: never step past it on +X.
                ux_raw = self._sign(tdx)
                ux = 0 if (ux_raw > 0 and ax + self.unit > self.half_line) else ux_raw
                uy = 0 if abs(tdy) <= self.unit else self._sign(tdy)
                return ux, uy
            else:
                # Not in strike range yet → converge to puck (still respect half-line)
                ux = 0 if (ax + self.unit > self.half_line) else self._sign(dx)
                uy = 0 if abs(dy) <= self.unit else self._sign(dy)
                return ux, uy

        # 2) No above receiver: if puck is heading downward, flip it upward
        if pvx < 0:
            # Make it non-dangerous in X first (flip to outbound), neutral Y
            ux = 0 if (ax + self.unit > self.half_line) else self._sign(dx)
            if ux == -1:
                uy = self.down_y_advance_safe(dy, abs(dy))
            else:
                uy = 0 if abs(dy) <= self.unit else self._sign(dy)
            return ux, uy

        # 3) Otherwise: light converge (supportive)
        ux = 0 if (ax + self.unit > self.half_line) else self._sign(dx)
        if ux == -1:
            uy = self.down_y_advance_safe(dy, abs(dy))
        else:
            uy = 0 if abs(dy) <= self.unit else self._sign(dy)
        yn = self._multi_teammate_y_nudge(ax, ay)
        if yn:
            uy = yn
        return ux, uy
