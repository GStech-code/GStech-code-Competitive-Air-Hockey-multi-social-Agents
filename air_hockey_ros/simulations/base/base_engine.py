from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import math
import random
from .rolling_queues import RollingMinMaxQueue, RollingAvgQueue

Command = Tuple[int, int, int]  # (agent_id, vx, vy)

class BaseEngine:
    width: int
    height: int
    tick_period: float  # seconds per tick (GameManager targets ~1/60)
    unit_speed_px: float  # maps int command (e.g., 1) to pixels/tick
    paddle_radius: float
    puck_radius: float
    friction_per_tick: float  # 1.0 = none
    bounce_damping: float
    halfline_policy: str  # "allow" | "clamp" | "soft"
    goal_gap_half: float  # scoring: vertical side gap half-height
    goal_semicircle_radius: float  # optional: for paddle-in-goal flags (no scoring effect now)
    jitter_enabled: bool
    jitter_seed: Optional[int]
    hold_last_ticks: int  # grace window for missed commands

    # --- State (mutable) ---
    tick: int
    scores_a: int
    scores_b: int
    agent_x: List[float]
    agent_y: List[float]
    agent_vx: List[float]
    agent_vy: List[float]
    agent_team: List[int]  # 0 = A, 1 = B (flat indexing: A first, then B)
    agent_last_cmd_tick: List[int]
    puck_x: float
    puck_y: float
    puck_vx: float
    puck_vy: float

    # --- Internal, physics helpers need these scratch buffers ---
    _prev_agent_x: List[float]
    _prev_agent_y: List[float]
    _rng: random.Random

    def __init__(self,) -> None:

        self.tick = 0
        self.scores_a = 0
        self.scores_b = 0

        self.agent_x = []
        self.agent_y = []
        self.agent_vx = []
        self.agent_vy = []
        self.agent_team = []
        self.agent_last_cmd_tick = []

        self.puck_x = 0.0
        self.puck_y = 0.0
        self.last_puck_x = self.puck_x
        self.last_puck_y = self.puck_y
        self.puck_vx = 0.0
        self.puck_vy = 0.0

        self._prev_agent_x = []
        self._prev_agent_y = []

        self.stuck_window = 1

        self.puck_x_history = RollingMinMaxQueue(self.stuck_window)
        self.puck_y_history = RollingMinMaxQueue(self.stuck_window)
        self.puck_velocity_history = RollingAvgQueue(self.stuck_window)

        self.jitter_seed = None
        self._rng = random.Random()

    def configure(self, **params) -> None:
        self.width = int(params.get("width", 800))
        self.height = int(params.get("height", 600))
        self.step_size = float(params.get("step_size", 1.0 / 60.0))
        self.unit_speed_px = float(params.get("unit_speed_px", 4.0))
        self.paddle_radius = float(params.get("paddle_radius", 20.0))
        self.puck_radius = float(params.get("puck_radius", 12.0))
        self.friction_per_tick = float(params.get("friction_per_tick", 1.0))
        self.bounce_damping = float(params.get("bounce_damping", 1.0))
        self.halfline_policy = str(params.get("halfline_policy", "allow"))  # "allow"|"clamp"|"soft"
        self.goal_gap_half = float(params.get("goal_gap_half", 120.0))
        self.goal_semicircle_radius = float(params.get("goal_semicircle_radius", self.width / 10.0))
        self.jitter_enabled = bool(params.get("jitter_enabled", True))
        self.jitter_seed = params.get("jitter_seed", None)
        self.hold_last_ticks = int(params.get("hold_last_ticks", 2))
        self.stuck_window = int(params.get("stuck_window", 60))
        self.puck_max_speed = float(params.get("puck_max_speed", 6.0))
        self.stuck_px_boundary = float(params.get("stuck_px_boundary", 20.0))

        if self.jitter_seed is not None:
            self._rng = random.Random(self.jitter_seed)

    def reset(self, num_agents_team_a: int, num_agents_team_b: int,
              agent_positions: Optional[List[Tuple[float, float]]] = None,
              puck_pos: Optional[Tuple[float, float]] = None,
              puck_vel: Optional[Tuple[float, float]] = None) -> None:
        # You’ll wire placements; physics doesn’t require full details here
        n = int(num_agents_team_a) + int(num_agents_team_b)
        self.agent_x = [0.0] * n
        self.agent_y = [0.0] * n
        self.agent_vx = [0.0] * n
        self.agent_vy = [0.0] * n
        self.agent_team = [0] * int(num_agents_team_a) + [1] * int(num_agents_team_b)
        self.agent_last_cmd_tick = [-10**9] * n  # far in the past
        self.puck_x_history.clear_and_re_capacity(self.stuck_window)
        self.puck_y_history.clear_and_re_capacity(self.stuck_window)
        self.puck_velocity_history.clear_and_re_capacity(self.stuck_window)


        # default placements if none provided (A at 1/4 width, B at 3/4)
        if agent_positions and len(agent_positions) == n:
            for i, (ax, ay) in enumerate(agent_positions):
                self.agent_x[i] = float(ax)
                self.agent_y[i] = float(ay)
        else:
            if num_agents_team_a > 0:
                gap_a = self.height / (num_agents_team_a + 1)
                for i in range(num_agents_team_a):
                    self.agent_x[i] = self.width * 0.25
                    self.agent_y[i] = gap_a * (i + 1)
            if num_agents_team_b > 0:
                gap_b = self.height / (num_agents_team_b + 1)
                for j in range(num_agents_team_b):
                    idx = num_agents_team_a + j
                    self.agent_x[idx] = self.width * 0.75
                    self.agent_y[idx] = self.height - gap_b * (j + 1)

        if puck_pos:
            self.puck_x, self.puck_y = float(puck_pos[0]), float(puck_pos[1])
        else:
            self.puck_x, self.puck_y = self.width / 2.0, self.height / 2.0

        if puck_vel:
            self.puck_vx, self.puck_vy = float(puck_vel[0]), float(puck_vel[1])
        else:
            self.puck_vx = 0.0
            self.puck_vy = 0.0

        self.last_puck_x = self.puck_x
        self.last_puck_y = self.puck_y
        self.scores_a = 0
        self.scores_b = 0
        self.tick = 0

    # --- Control & stepping ---
    def apply_commands(self, commands: List[Command]) -> None:
        """
        commands: [(agent_id, vx, vy)]
        - vx, vy are raw ints (e.g., -1/0/1; -2/2 later). No validation/clamp here.
        - Unmentioned agents keep their last command until hold_last_ticks expires.
        """
        now = self.tick
        for aid, vx, vy in commands:
            # store raw command; integration scales by unit_speed_px
            self.agent_vx[aid] = float(vx) * self.unit_speed_px
            self.agent_vy[aid] = float(vy) * self.unit_speed_px
            self.agent_last_cmd_tick[aid] = now

    def step(self) -> None:
        """
        Advance one simulation tick. Order matters:
        1) Apply command policy (hold-last, timeout)
        2) Integrate agents
        3) Enforce world bounds and half-line policy
        4) Puck-agent collisions
        5) Integrate puck (includes friction)
        6) Puck-wall/goal interactions (may score + faceoff)
        """
        self._apply_command_policy()
        self._integrate_agents()
        self._enforce_bounds_and_halfline()
        self._collide_puck_agents()
        self._integrate_puck()
        self._puck_walls_and_goals()
        self._detect_and_reset_if_stuck()
        self.tick += 1

    # --- Queries ---
    def get_world_state(self) -> Dict:
        """
        Return a COPY of world state (ROS/GameManager logging format).
        """
        return {
            "puck_x": self.puck_x,
            "puck_y": self.puck_y,
            "puck_vx": self.puck_vx,
            "puck_vy": self.puck_vy,
            "agent_x": self.agent_x[:],
            "agent_y": self.agent_y[:],
            "agent_vx": self.agent_vx[:],
            "agent_vy": self.agent_vy[:],
        }

    def get_scores(self) -> Dict:
        return {"team_a_score": self.scores_a, "team_b_score": self.scores_b}

    # --- Internals (kept minimal; hot-path helpers) ---
    def _apply_command_policy(self) -> None:
        """
        Hold-last policy with timeout.
        If an agent hasn't received a command within hold_last_ticks, zero its velocity.
        Assumes self.agent_vx/vy are set by apply_commands when commands arrive.
        """
        if self.hold_last_ticks <= 0:
            return
        n = len(self.agent_x)
        now = self.tick
        for i in range(n):
            if now - self.agent_last_cmd_tick[i] > self.hold_last_ticks:
                # timeout -> stop (no decay for now; simple and deterministic)
                self.agent_vx[i] = 0.0
                self.agent_vy[i] = 0.0

    def _integrate_agents(self) -> None:
        """
        Move agents by their commanded velocities scaled to pixels per tick.
        Stores previous positions for collision response (actual paddle velocity).
        """
        n = len(self.agent_x)
        self._prev_agent_x = [0.0] * n
        self._prev_agent_y = [0.0] * n

        for i in range(n):
            self._prev_agent_x[i] = self.agent_x[i]
            self._prev_agent_y[i] = self.agent_y[i]
            self.agent_x[i] += self.agent_vx[i]
            self.agent_y[i] += self.agent_vy[i]

    def _enforce_bounds_and_halfline(self) -> None:
        """
        Keep agents inside the rink; apply half-line policy.
        - "clamp": hard restrict to own half
        - "allow": no half restriction
        - "soft": no clamp (you can later emit a violation flag if you want)
        """
        r = self.paddle_radius
        w = self.width
        h = self.height
        mid = w / 2.0
        policy = self.halfline_policy

        for i in range(len(self.agent_x)):
            # Walls (full rink)
            if self.agent_y[i] < r:
                self.agent_y[i] = r
            elif self.agent_y[i] > h - r:
                self.agent_y[i] = h - r

            if self.agent_x[i] < r:
                self.agent_x[i] = r
            elif self.agent_x[i] > w - r:
                self.agent_x[i] = w - r

            if policy == "clamp":
                if self.agent_team[i] == 0:  # Team A (left)
                    if self.agent_x[i] > mid - r:
                        self.agent_x[i] = mid - r
                else:  # Team B (right)
                    if self.agent_x[i] < mid + r:
                        self.agent_x[i] = mid + r
            # "allow" and "soft": no clamp here

    def _collide_puck_agents(self) -> None:
        """
        Disc <-> Paddle collision adapted from your Disc.handle_paddle_collision:
        - Compute normal from paddle to puck
        - Reflect relative velocity along normal with restitution < 1
        - Add a fraction of paddle 'actual' velocity (based on displacement this tick)
        - Separate overlap
        - Cap puck speed
        - Optional tiny jitter for liveliness
        """
        pr = self.puck_radius
        ar = self.paddle_radius
        total_r = pr + ar

        # Constants ported from your Disc
        disc_mass = 1.0
        paddle_mass = 3.0
        restitution = 0.75
        velocity_transfer = 0.3

        n = len(self.agent_x)
        for i in range(n):
            dx = self.puck_x - self.agent_x[i]
            dy = self.puck_y - self.agent_y[i]
            dist2 = dx * dx + dy * dy
            if dist2 > (total_r * total_r):
                continue

            # Compute normal
            dist = math.sqrt(dist2) if dist2 > 0.0 else 0.0
            if dist == 0.0:
                nx, ny = 1.0, 0.0
                dist = 1.0
            else:
                nx = dx / dist
                ny = dy / dist

            # Paddle "actual" velocity from displacement this tick
            pvx = self.agent_x[i] - self._prev_agent_x[i]
            pvy = self.agent_y[i] - self._prev_agent_y[i]

            # Relative velocity (disc minus paddle)
            rvx = self.puck_vx - pvx
            rvy = self.puck_vy - pvy

            # Relative velocity along normal
            rvn = rvx * nx + rvy * ny
            if rvn > 0.0:
                # Separating, no impulse
                # Still ensure separation to avoid sticking if overlapping
                pass
            else:
                # Collision impulse (disc vs heavier paddle)
                mass_ratio = (2.0 * paddle_mass) / (disc_mass + paddle_mass)
                impulse = -(1.0 + restitution) * rvn * mass_ratio
                self.puck_vx += impulse * nx
                self.puck_vy += impulse * ny

                # Add some of paddle velocity influence
                self.puck_vx += pvx * velocity_transfer
                self.puck_vy += pvy * velocity_transfer

            # Positional correction (separate overlap)
            overlap = total_r - dist
            if overlap > 0.0:
                self.puck_x += nx * (overlap + 1e-6)
                self.puck_y += ny * (overlap + 1e-6)

            # Cap puck speed
            speed = math.hypot(self.puck_vx, self.puck_vy)
            if speed > self.puck_max_speed:
                scale = self.puck_max_speed / speed
                self.puck_vx *= scale
                self.puck_vy *= scale

            # Tiny jitter (configurable)
            if self.jitter_enabled:
                self.puck_vx += self._rng.uniform(-0.2, 0.2)
                self.puck_vy += self._rng.uniform(-0.2, 0.2)

    def _integrate_puck(self) -> None:
        """
        Move puck by its current velocity; apply per-tick friction.
        Matches Disc.update + optional multiplicative friction.
        """
        self.puck_x += self.puck_vx
        self.puck_y += self.puck_vy

        if self.friction_per_tick != 1.0:
            self.puck_vx *= self.friction_per_tick
            self.puck_vy *= self.friction_per_tick

        # Top/bottom walls (simple elastic with damping factor)
        r = self.puck_radius
        if self.puck_y < r:
            self.puck_y = r
            self.puck_vy = -self.puck_vy * self.bounce_damping
        elif self.puck_y > self.height - r:
            self.puck_y = self.height - r
            self.puck_vy = -self.puck_vy * self.bounce_damping

    def _puck_walls_and_goals(self) -> None:
        """
        Handle left/right boundaries as walls with a central scoring gap on each side.
        Ported from Disc.check_side_collision with optional damping.
        On goal: increment score, center faceoff with gentle push toward conceding side.
        """
        r = self.puck_radius
        # Left boundary
        if self.puck_x < r:
            # scoring gap on the left side refers to the *left goal mouth*
            gap_lo = (self.height / 2.0) - self.goal_gap_half
            gap_hi = (self.height / 2.0) + self.goal_gap_half
            if gap_lo <= self.puck_y <= gap_hi:
                # Team B scores (right team)
                self.scores_b += 1
                self._center_faceoff(direction=+1)
                return
            else:
                self.puck_x = r
                self.puck_vx = -self.puck_vx * self.bounce_damping

        # Right boundary
        if self.puck_x > self.width - r:
            gap_lo = (self.height / 2.0) - self.goal_gap_half
            gap_hi = (self.height / 2.0) + self.goal_gap_half
            if gap_lo <= self.puck_y <= gap_hi:
                # Team A scores (left team)
                self.scores_a += 1
                self._center_faceoff(direction=-1)
                return
            else:
                self.puck_x = self.width - r
                self.puck_vx = -self.puck_vx * self.bounce_damping

    def _detect_and_reset_if_stuck(self) -> None:
        """
        Detect puck stuck in a small area (e.g. trapped between paddles).
        Uses last 60 positions. If bounding box is too small and puck
        keeps moving, treat as stuck and reset to center.
        """
        # Record position
        self.puck_x_history.append(self.puck_x)
        self.puck_y_history.append(self.puck_y)
        self.puck_velocity_history.append(math.hypot(self.puck_x - self.last_puck_x, self.puck_y - self.last_puck_y))
        self.last_puck_x = self.puck_x
        self.last_puck_y = self.puck_y

        if len(self.puck_velocity_history) < self.stuck_window:
            return  # not enough data yet

        # Heuristic thresholds
        if (self.puck_x_history.range() < self.stuck_px_boundary
                and self.puck_y_history.range() < self.stuck_px_boundary
                and self.puck_velocity_history.avg() > 0.1):
            # Reset puck to center without scoring
            self._center_faceoff()
            # Clear history after reset
            self.puck_x_history.clear()
            self.puck_y_history.clear()
            self.puck_velocity_history.clear()

    # ====== small helper ======

    def _center_faceoff(self, direction: int = 0) -> None:
        """
        Reset puck to center and nudge toward the side that conceded.
        direction: +1 -> toward right, -1 -> toward left
        """
        self.puck_x = self.width / 2.0
        self.puck_y = self.height / 2.0
        # Serve angle: keep it simple & deterministic here; jitter (if enabled) will add small variation
        base_speed = 3.0
        self.puck_vx = base_speed * float(direction)
        self.puck_vy = 0.0
