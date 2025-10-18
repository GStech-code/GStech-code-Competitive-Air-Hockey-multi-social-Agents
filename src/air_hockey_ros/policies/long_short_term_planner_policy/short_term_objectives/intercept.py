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

    def __init__(self, agent_id, teammate_ids, commands, rules: Dict, **params):
        super().__init__(agent_id, teammate_ids, commands, rules, **params)

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

        # Direction-change detection (reuse FastShot style)
        self.last_pvx, self.last_pvy = 0.0, 0.0
        self.change_tol = math.cos(0.1)

        # State
        self.assume_intercept = False

    # ----------------- Helpers (mirroring FastShot) -----------------

    @staticmethod
    def _sign(v: float) -> int:
        return 0 if v == 0 else (1 if v > 0 else -1)

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
            return 0
        nxt = exp_x + self.unit
        return 0 if nxt > self._half_line else 1

    def _guard_halfline_twice(self, exp_x: float) -> Tuple[int, int]:
        """Two-step variant; snap back if already over the line."""
        if exp_x >= self._half_line:
            return -1, 0
        exp_x += self.unit
        if exp_x >= self._half_line:
            return 0, 0
        exp_x += self.unit
        if exp_x >= self._half_line:
            return 1, 0
        return 1, 1

    def puck_dir_changed(self, pvx, pvy):
        # Both static → no change
        if ((abs(pvx) < 1e-3 and abs(pvy) < 1e-3) and
            (abs(self.last_pvx) < 1e-3 and abs(self.last_pvy) < 1e-3)):
            return False
        # One static, other not → change
        if ((abs(pvx) < 1e-3 and abs(pvy) < 1e-3) or
            (abs(self.last_pvx) < 1e-3 and abs(self.last_pvy) < 1e-3)):
            return True
        dot = (pvx * self.last_pvx + pvy * self.last_pvy) / (
            ((pvx ** 2 + pvy ** 2) * (self.last_pvx ** 2 + self.last_pvy ** 2)) ** 0.5 + 1e-9)
        return dot < self.change_tol

    def _project_y_with_wall_once(self, py: float, pvy: float, steps: int) -> float:
        """
        One-bounce vertical projection for short horizon:
        - Move to py + pvy*steps.
        - If passes a wall once, reflect around that wall and stop.
        """
        raw = py + pvy * steps
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

    def handle_intercept(self) -> bool:
        """
        Decide if our 'we will/should intercept' assumption is actually true.
        Assumptions (as per your rules of engagement):
          - Puck direction has NOT changed (checked outside).
          - We believed a hit should have happened soon or will happen with current queue.

        Returns False  -> it's a MISS (we won't/didn't hit under current queue & physics)
                True -> not a miss (a hit either already occurred within tolerance,
                                    or is still feasible within the short horizon),
                                    interception commands handled

        Robustness vs. naive 'future update':
          - Simulates step-by-step, applying our queued commands.
          - Detects continuous-time collision within each tick (not just gridpoint overlap).
          - Handles Y-wall reflections.
          - Early-outs on clear geometric divergence.
        """
        # --- constants / geometry ---
        unit = self.unit
        pr = self.paddle_radius
        r = pr + self.puck_radius  # contact radius
        r2 = (r + 0.5 * unit) ** 2  # light tolerance
        half = self._half_line
        low_wall_puck, up_wall_puck = self._low_wall_puck, self._up_wall_puck

        # --- read state ---
        ws = self.last_ws
        ax = ws["agent_x"][self.agent_id]
        ay = ws["agent_y"][self.agent_id]
        px = ws["puck_x"]
        py = ws["puck_y"]
        pvx = ws["puck_vx"]
        pvy = ws["puck_vy"]

        # --- snapshot queued commands (so we don't race with concurrent pushes) ---
        planned = self.commands.peek_all()  # expects (ux,uy) tuples; non-destructive

        # --- helpers ---
        def min_dist_sq_segment_to_point(p0x, p0y, p1x, p1y) -> float:
            """Distance^2 of a segment in RELATIVE motion to the origin (0,0).
            We'll pass the relative segment (puck - paddle) for the current tick."""
            vx = p1x - p0x
            vy = p1y - p0y
            denom = vx * vx + vy * vy
            if denom <= 1e-12:
                # Segment is essentially a point
                return p0x * p0x + p0y * p0y
            t = - (p0x * vx + p0y * vy) / denom
            if t <= 0.0:
                qx, qy = p0x, p0y
            elif t >= 1.0:
                qx, qy = p1x, p1y
            else:
                qx, qy = (p0x + t * vx), (p0y + t * vy)
            return qx * qx + qy * qy

        # --- seed previous positions for swept test ---
        prev_ax, prev_ay = ax, ay
        prev_px, prev_py = px, py

        # Quick coarse check: if puck is inbound and our forward progress is clamped below half-line,
        # and the relative X distance is growing for a couple of steps, we can early-out as miss.
        growing_divergence = 0

        # --- simulate up to 'horizon' ticks ---
        for step in range(1, len(planned) + 1):
            # Apply our command for this tick (if any), with half-line guard.
            ux, uy = (planned[step - 1] if step - 1 < len(planned) else (0, 0))

            # Half-line guard: forbid crossing beyond half-line on +x moves.
            if ux > 0 and ax + unit > half:
                ux = 0

            # Advance paddle
            ax_next = ax + ux * unit
            ay_next = ay + uy * unit

            # Advance puck with possible Y reflections
            px_next = px + pvx
            py_next = self._project_y_with_wall_once(py, pvy, 1)

            # --- continuous collision check for this tick via relative segment ---
            # Relative positions (puck - paddle), from t to t+1
            rel0x, rel0y = (prev_px - prev_ax), (prev_py - prev_ay)
            rel1x, rel1y = (px_next - ax_next), (py_next - ay_next)

            d2 = min_dist_sq_segment_to_point(rel0x, rel0y, rel1x, rel1y)
            if d2 <= r2:
                # Contact within this tick; not a miss
                adv_x, adv_y = self.commands.get_advance()
                exp_x = ax + adv_x * self.unit
                exp_y = ay + adv_y * self.unit
                ux = -1 if exp_x > self.x_home else 0
                lane_y = self.y_center
                uy = 0 if abs(lane_y - exp_y) <= self.unit else self._sign(lane_y - exp_y)
                yn = self._teammate_y_nudge(ws, exp_x, exp_y)
                if yn != 0:
                    uy = yn
                self.commands.push((ux, uy))
                if ux == 0 and uy == 0:
                    self.long_term_mode()
                return True

            # --- coarse divergence heuristic (helps early-out) ---
            # If puck is inbound (pvx <= 0) but our +x progress is clamped (ux==0 and ax<half),
            # and the relative X distance increased, count it; two consecutive growths => miss soon.
            inbound = pvx <= 0
            relx_prev = (prev_px - prev_ax)
            relx_curr = (px_next - ax_next)
            if inbound and ux == 0 and ax < half and (relx_curr > relx_prev + 0.5 * unit):
                growing_divergence += 1
            else:
                growing_divergence = 0
            if growing_divergence >= 2:
                return False

            # roll forward
            prev_ax, prev_ay = ax_next, ay_next
            prev_px, prev_py = px_next, py_next
            ax, ay = ax_next, ay_next
            px, py = px_next, py_next

        # If we exhausted the horizon with no continuous contact, treat as miss.
        return False

    # ----------------- Main (FastShot-like) -----------------

    def emergency_step(self, ws: Dict):
        """
        Called after an offensive objective; queue is empty.
        Must issue the *first* command fast, and end with ~2–3 total commands.
        Next step will always be new_ws_step().
        """
        ay = ws["agent_y"][self.agent_id]
        py = ws["puck_y"]
        pvy = ws["puck_vy"]

        fut_py = py + pvy
        if fut_py <= ay - self.paddle_radius:
            self.commands.push((-1, -1))
        elif fut_py >= ay + self.paddle_radius:
            self.commands.push((-1, 1))
        else:
            self.commands.push((-1, 0))

        self.intro_step(ws)

    def intro_step(self, ws: Dict):
        self.last_ws = ws
        self.assume_intercept = False
        self.last_pvx, self.last_pvy = ws["puck_vx"], ws["puck_vy"]
        self.continue_step()

    def new_ws_step(self, ws: Dict):
        self.last_ws = ws
        pvx = ws["puck_vx"]
        pvy = ws["puck_vy"]

        if not self.commands.is_empty():
            if self.puck_dir_changed(pvx, pvy):
                self.commands.clear()
                self.assume_intercept = False
            elif self.assume_intercept:
                if self.handle_intercept():
                    return
                self.commands.clear()
                self.assume_intercept = False
        # Update last dir
        self.last_pvx, self.last_pvy = pvx, pvy
        self.continue_step()

    def continue_step(self):
        ws = self.last_ws

        # --- Unpack ---
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

        # Short-horizon projection (when our next push lands) + single wall reflection
        future_steps = self.commands.get_size() + 1
        fut_px = px + pvx * future_steps
        fut_py = self._project_y_with_wall_once(py, pvy, future_steps)

        # ===================== A) Outbound puck (avoid contact) =====================
        dx = fut_px - exp_x
        dy = fut_py - exp_y
        ady = abs(dy)
        if pvx > 0:

            # If we'd collide soon, sidestep away from puck Y and mildly retreat X
            if (abs(dx) <= self._hit_d2) and (ady <= self._hit_d2):
                ux = -1 if exp_x > self.paddle_radius else 0
                # Your requested dodge rule: move away from puck Y
                uy = 1 if dy < 0 else -1
                self.commands.push_multiple([(ux, uy), (ux, uy)])
                return

            # Otherwise: posture toward defensive home when puck not close to our half
            # "Not close to our half": future puck x beyond half-line margin.
            if fut_px > self._half_line:
                dxh = self.x_home - exp_x
                ux = 0 if abs(dxh) <= self.unit else self._sign(dxh)
            else:
                # drift to x_home
                ux = -1 if exp_x > self.x_home else 0
            dyh = self.y_center - exp_y
            uy = 0 if abs(dyh) <= self.unit else self._sign(dyh)
            yn = self._teammate_y_nudge(ws, exp_x, exp_y)
            if yn != 0:
                uy = yn
            self.commands.push_multiple([(ux, uy), (ux, uy)])
            if ux == 0 and uy == 0:
                self.long_term_mode()
            return

        # ===================== B) Inbound puck (intercept earliest safe) =====================


        # If behind us: basic back-hit safety first (don’t let it clip from behind)
        r_dx = exp_x - fut_px
        if fut_px < exp_x and (r_dx <= self._hit_d2) and (ady <= self._hit_d2):
            if r_dx <= self._hit_d and ady >= self._hit_d:
                self.commands.push((-1, 0))
                self.long_term_mode()
                return
            ux1, ux2 = self._guard_halfline_twice(exp_x)
            # Move away from incoming puck Y (mirror your outbound dodge; protects from self-hit)
            uy = 1 if dy < 0 else -1
            self.commands.push_multiple([(ux1, uy), (ux2, uy)])
            return

        # Choose minimal-move burst toward projected contact, with half-line guard
        ux = 0 if abs(dx) <= self.unit else self._sign(dx)
        if ux > 0:
            ux = self._guard_halfline(exp_x)  # forbid crossing half
        uy = 0 if abs(dy) <= self.unit else self._sign(dy)

        # Teammate lane overlap → nudge Y
        yn = self._teammate_y_nudge(ws, exp_x, exp_y)
        if yn != 0:
            uy = yn

        self.commands.push_multiple([(ux, uy), (ux, uy)])

        # -------- Assume-intercept management (your point C) --------
        # Start assuming only if we're truly converging to contact soon
        close_next = (abs(dx) <= self._hit_d2) and (abs(dy) <= self._hit_d2)
        if close_next and (ux != 0 or uy != 0):
            self.assume_intercept = True
