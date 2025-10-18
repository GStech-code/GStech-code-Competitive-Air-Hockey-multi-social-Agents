# pass_shot.py
from typing import Dict
from .objective import Objective
import math

class Pass_Shot(Objective):
    """
    M1: Pass
    Purpose: Feed the puck toward a teammate when safe and meaningful.
    - No params, only internal factors.
    - Avoids overlapping teammates using stronger multi-nudge logic.
    - Positions near the center band, drifts dynamically when idle.
    """

    def __init__(self, agent_id, teammate_ids, commands, rules: Dict, **params):
        super().__init__(agent_id, teammate_ids, commands, rules, **params)
        self.width = rules.get("width", 800)
        self.height = rules.get("height", 600)
        self.unit = rules.get("unit_speed_px", 4.0)
        self.paddle_r = rules.get("paddle_radius", 20)
        self.puck_r = rules.get("puck_radius", 12)
        self.double_r = self.paddle_r * 2
        self.quadruple_r = self.double_r * 2
        self.y_center = self.height / 2.0
        self.half_line = self.width * 0.5
        self.hold_line = self.half_line - self.paddle_r - self.unit
        self.half_unit = self.unit * 0.5
        self.x_home = self.width * 0.45  # central stance, slightly defensive

        # factors
        self.teamwork_bias = params.get("teamwork_bias", 0.7)
        self.risk_tolerance = params.get("risk_tolerance", 0.4)
        self.pass_accuracy = params.get("pass_accuracy", 0.8)

        # short memory
        self.last_pvx, self.last_pvy = 0.0, 0.0
        self.change_tol = math.cos(0.1)
        self._hit_d2 = self.paddle_r + self.puck_r + (self.unit * 2)
        self._low_wall = self.puck_r
        self._up_wall = self.height - self.puck_r

    # ----------------- Helpers -----------------

    def _sign(self, v: float) -> int:
        return 0 if abs(v) < 1e-6 else (1 if v > 0 else -1)

    def _multi_teammate_y_nudge(self, exp_x: float, exp_y: float) -> int:
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
            dx = abs(tx - exp_x)
            dy = ty - exp_y
            # influence only if close enough in X
            if dx < self.quadruple_r:
                weight = 1.0 / (1.0 + dx / self.double_r)
                total_force += -math.copysign(weight, dy)
        if abs(total_force) < 0.2:
            return 0
        return 1 if total_force > 0 else -1

    def _project_y_with_wall_once(self, py: float, pvy: float, steps: int) -> float:
        """Simple single-bounce projection, identical to others."""
        raw = py + pvy * steps
        if self._low_wall <= raw <= self._up_wall:
            return raw
        if raw > self._up_wall:
            over = raw - self._up_wall
            return self._up_wall - over
        if raw < self._low_wall:
            over = self._low_wall - raw
            return self._low_wall + over
        return raw

    def puck_dir_changed(self, pvx, pvy) -> bool:
        if (abs(pvx) < 1e-3 and abs(pvy) < 1e-3) and \
           (abs(self.last_pvx) < 1e-3 and abs(self.last_pvy) < 1e-3):
            return False
        if (abs(pvx) < 1e-3 and abs(pvy) < 1e-3) or \
           (abs(self.last_pvx) < 1e-3 and abs(self.last_pvy) < 1e-3):
            return True
        dot = (pvx * self.last_pvx + pvy * self.last_pvy) / (
            math.sqrt((pvx**2 + pvy**2) * (self.last_pvx**2 + self.last_pvy**2)) + 1e-9)
        return dot < self.change_tol
    # ----------------- Main -----------------

    def emergency_step(self, ws: Dict):
        """
        Called after an offensive objective; queue is empty.
        Must issue the *first* command fast, and end with ~2–3 total commands.
        Next step will always be new_ws_step().
        """
        self.a_x = ws["agent_x"]
        a_y = ws["agent_y"]
        self.a_y = a_y
        ay = a_y[self.agent_id]
        py = ws["puck_y"]
        pvy = ws["puck_vy"]

        fut_py = py + pvy
        if fut_py <= ay - self.paddle_r:
            uy = -1
        elif fut_py >= ay + self.paddle_r:
            uy = 1
        else:
            uy = 0

        self.commands.push((0, uy))

        self.intro_step(ws)

    def intro_step(self, ws: Dict):
        self.last_ws = ws
        self.a_x = ws["agent_x"]
        self.a_y = ws["agent_y"]
        self.last_pvx, self.last_pvy = ws["puck_vx"], ws["puck_vy"]
        self.continue_step()

    def new_ws_step(self, ws: Dict):
        self.last_ws = ws
        self.a_x = ws["agent_x"]
        self.a_y = ws["agent_y"]
        pvx = ws["puck_vx"]
        pvy = ws["puck_vy"]

        if not self.commands.is_empty() and self.puck_dir_changed(pvx, pvy):
            self.commands.clear()

        # Update last dir
        self.last_pvx, self.last_pvy = pvx, pvy
        self.continue_step()

    def continue_step(self):
        """
        Maintain central supportive stance; attempt passes when conditions align.
        """
        ws = self.last_ws

        # --- Unpack ---
        a_x = self.a_x
        a_y = self.a_y
        ax = a_x[self.agent_id]
        ay = a_y[self.agent_id]
        px = ws["puck_x"]
        py = ws["puck_y"]
        pvx = ws["puck_vx"]
        pvy = ws["puck_vy"]


        # Expected after queued cmds
        adv_x, adv_y = self.commands.get_advance()
        exp_x = ax + adv_x * self.unit
        exp_y = ay + adv_y * self.unit

        # Short-horizon projection (when our next push lands) + single wall reflection
        future_steps = self.commands.get_size() + 1
        fut_px = px + pvx * future_steps
        fut_py = self._project_y_with_wall_once(py, pvy, future_steps)

        dx = fut_px - exp_x
        dy = fut_py - exp_y
        p_inbound = (pvx < 0)
        pvx_out = (pvx >= 0)  # outbound (to the right)
        pv_fast = (pvx > self.half_unit and pvy >= self.half_unit)

        # Closest receiver (self vs teammates)

        # ===== 1) Puck beyond hold line =====
        if fut_px > self.hold_line:
            if pvx_out:
                # (A) Outbound & already past hold: reposition/hold; don't chase
                ux = -1 if exp_x > self.x_home else 0
                uy = 0 if abs(self.y_center - exp_y) <= self.unit else self._sign(self.y_center - exp_y)
                yn = self._multi_teammate_y_nudge(exp_x, exp_y)
                if yn:
                    uy = yn
                self.commands.push_multiple([(ux, uy), (ux, uy)])
                return
            else:
                # (B) Inbound but already past hold: prep to defend/redirect (no half-line crossing)
                ux = 0  # never +x here
                uy = 0 if abs(fut_py - exp_y) <= self.unit else self._sign(fut_py - exp_y)
                yn = self._multi_teammate_y_nudge(exp_x, exp_y)
                if yn:
                    uy = yn
                self.commands.push_multiple([(ux, uy), (ux, uy)])
                return

        # ===== 2) Puck behind us in x (left of paddle) =====
        if fut_px < exp_x:
            # Distances in our frame
            adx = exp_x - fut_px  # how far behind in X
            ady = abs(dy)

            # Too close from behind → immediate avoidance: step AWAY in Y, never advance in X
            if (adx <= self._hit_d2) and (ady <= self._hit_d2):
                uy = 1 if dy < 0 else -1  # move away from puck's Y (no teammate override here)
                ux = -1 if exp_x > self.paddle_r else 0  # mild retreat; NEVER +x in this subcase
                self.commands.push_multiple([(ux, uy), (ux, uy)])
                return

            # Otherwise: do NOT back-chase; central hold with teammate respect
            ux = -1 if exp_x > self.x_home else 0
            uy = 0 if abs(dy) <= self.unit else self._sign(dy)
            yn = self._multi_teammate_y_nudge(exp_x, exp_y)
            if yn: uy = yn
            self.commands.push_multiple([(ux, uy), (ux, uy)])
            return

        # ===== 3) Fast inbound (both components meaningful) =====
        if pv_fast:
            # Stay supportive: no +x chase; gently align Y; stronger teammate yield
            ux = -1 if exp_x > self.x_home else 0  # at most retreat, never advance here
            uy = 0 if abs(fut_py - exp_y) <= self.unit else self._sign(fut_py - exp_y)
            yn = self._multi_teammate_y_nudge(exp_x, exp_y)
            if yn:
                uy = yn
            self.commands.push_multiple([(ux, uy), (ux, uy)])
            return

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
            ux = -1 if exp_x > self.x_home else 0
            uy = self._sign(self.y_center - exp_y) if abs(self.y_center - exp_y) > self.unit else 0
            yn = self._multi_teammate_y_nudge(exp_x, exp_y)
            if yn:
                uy = yn
            self.commands.push_multiple([(ux, uy), (ux, uy)])
            return

        # ===== 5) Default pass =====
        # Try to PASS to nearest teammate above us; else kill downward motion; else light converge

        # 1) Nearest ABOVE teammate (ty < exp_y)
        recv = None
        best = 1e18
        for tid in self.teammate_ids:
            tx, ty = a_x[tid], a_y[tid]
            if tx > exp_x:  # "above" == ahead toward enemy goal
                d2 = (tx - exp_x) ** 2 + (ty - exp_y) ** 2
                if d2 < best:
                    best, recv = d2, (tx, ty)

        if recv:
            tx, ty = recv

            # Step 1: converge to puck (paddle can't cross half-line)
            u1x = 0 if (exp_x + self.unit > self.half_line) else self._sign(fut_px - exp_x)
            u1y = 0 if abs(fut_py - exp_y) <= self.unit else self._sign(fut_py - exp_y)

            # Step 2: strike from puck toward receiver
            vx, vy = (tx - fut_px), (ty - fut_py)

            # X: keep/raise pvx, never reduce it (no negative strike)
            if pvx >= 0:
                u2x = 0 if pvx >= self.half_unit else 1  # don't over-accelerate if already decent
            else:
                u2x = 1  # flip inbound to outbound

            # Still respect paddle half-line on the second step
            if u2x > 0 and (exp_x + self.unit > self.half_line):
                u2x = 0

            # Y: aim toward teammate vertically
            u2y = self._sign(vy)

            self.commands.push_multiple([(u1x, u1y), (u2x, u2y)])
            return

        # 2) No above receiver: if puck is heading downward, flip it upward
        if pvx < 0:
            # Make it non-dangerous in X first (flip to outbound), neutral Y
            u1x = 0 if (exp_x + self.unit > self.half_line) else self._sign(dx)
            u1y = 0 if abs(dy) <= self.unit else self._sign(dy)
            u2x = 1  # push to outbound
            if (exp_x + self.unit > self.half_line):
                u2x = 0
            u2y = 0
            self.commands.push_multiple([(u1x, u1y), (u2x, u2y)])
            return

        # 3) Otherwise: light converge (supportive)
        ux = 0 if (exp_x + self.unit > self.half_line) else self._sign(dx)
        uy = 0 if abs(dy) <= self.unit else self._sign(dy)
        yn = self._multi_teammate_y_nudge(exp_x, exp_y)
        if yn:
            uy = yn
        self.commands.push_multiple([(ux, uy), (ux, uy)])

