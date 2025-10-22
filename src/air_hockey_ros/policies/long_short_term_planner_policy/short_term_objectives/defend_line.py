from typing import Dict, List, Tuple, Optional
from enum import IntEnum
from .objective import Objective

class DefendLine(Objective):
    def __init__(self, agent_id, teammate_ids, rules: Dict, **params):
        super().__init__(agent_id, teammate_ids, rules, **params)
        self.paddle_radius = rules.get('paddle_radius', 20)
        self.unit_speed_px = rules.get('unit_speed_px', 4.0)
        self.y_center = rules.get('height', 600) / 2.0
        self.limit_x = rules.get('width', 800) / 2 - self.paddle_radius
        puck_radius = rules.get('puck_radius', 12)
        self._hit_d = self.paddle_radius + puck_radius + self.unit_speed_px
        self._hit_d2 = self._hit_d + self.unit_speed_px
        self.paddle_distance = self.paddle_radius * 2 + self.unit_speed_px
        self.puck_max_speed = rules.get('puck_max_speed', 6.0)
        self.puck_third_speed = self.puck_max_speed / 3.0
        row = params.get('defense_row', rules.get('goal_offset', 40) + self.paddle_radius + self.unit_speed_px)
        half_unit = self.unit_speed_px / 2
        self.min_x_band = row - half_unit
        self.max_x_band = row + half_unit
        self.min_puck_underdrift = max(0, self.min_x_band - self._hit_d - self.unit_speed_px)
        self.min_encounter = self.min_x_band - self.paddle_radius
        self.max_encounter = self.max_x_band + self.paddle_radius
        threat_multiplier = params.get('threat_multiplier', 2)
        self.puck_threat_distance = self._hit_d * threat_multiplier


    @staticmethod
    def _sign(v: float) -> int:
        return 0 if abs(v) < 1e-6 else (1 if v > 0 else -1)

    def down_y_advance_safe(self, dy, ady):
        if ady > self._hit_d2:
            return self._sign(dy)
        elif ady == self._hit_d2:
            return 0
        return self._sign(-dy)

    def step(self, ws: Dict) -> Tuple[int, int]:
        """
        Reactive update when a new world state arrives.
        Lightweight, geometry-aware
        """
        # --- basic unpacking ---
        agent_x = ws['agent_x']
        agent_y = ws['agent_y']
        ax = agent_x[self.agent_id]
        ay = agent_y[self.agent_id]
        px = ws["puck_x"]
        py = ws["puck_y"]
        pvx = ws["puck_vx"]
        pvy = ws["puck_vy"]

        unit_speed_px = self.unit_speed_px
        hit_d = self._hit_d
        fut_px = px + pvx
        fut_py = py + pvy

        # --- quick threat checks ---
        dx = fut_px - ax
        adx = abs(dx)
        dy = fut_py - ay
        ady = abs(dy)

        if pvx > 0:
            if adx <= hit_d and ady <= self._hit_d2:
                uy = -1 if ady >= 0 else 1
                ux = 1 if (ax + unit_speed_px) <= self.limit_x else 0
                return ux, uy

            cdy = self.y_center - ay
            if abs(cdy) <= unit_speed_px:
                uy = 0
            else:
                uy = 1 if cdy > 0 else -1

            if ax < self.min_x_band:
                ux = 1
            elif ax > self.max_x_band:
                ux = -1
            else:
                ux = 0

            return ux, uy

        if dx < 0:
            if  dx <= self._hit_d2 and ady <= hit_d:
                ux = 1 if ax < self.limit_x else 0
                uy = 1 if ady <= 0 else -1
                return ux, uy
            ux = -1 if ax > self.min_puck_underdrift else 0
            uy = self.down_y_advance_safe(dy, ady)
            return ux, uy

        # --- X correction: move into band only if outside ---
        # --- Y correction ---
        if ax < self.min_x_band:
            ux = 1
        elif ax > self.max_x_band:
            ux = -1
        else:
            ux = 0

        # Choose an X intercept on our band near current paddle X
        x_i = min(max(ax + ux, self.min_x_band), self.max_x_band)
        # Time for puck to reach x_i (linear; pvx < 0 so t >= 0 when x_i < px)
        t = (x_i - px) / pvx if pvx != 0 else 0.0

        # Intercept Y estimate (simple linear; keeps it lean)
        tgt_y = py + pvy * t if t > 0 else fut_py

        # Y steer to the intercept (1 step granularity)
        err_y = tgt_y - ay
        uy = 0 if abs(err_y) <= unit_speed_px else (1 if err_y > 0 else -1)


        if fut_px - ax <= self.puck_threat_distance:
            teammate_ids = self.teammate_ids
            # if a teammate is very close in y within our band, nudge one step away
            if teammate_ids and self.min_x_band <= ax <= self.max_x_band:
                for tid in teammate_ids:
                    tx, ty = agent_x[tid], agent_y[tid]
                    if ((self.min_encounter < tx < self.max_encounter)
                        and abs(ty - ay) <= self.paddle_distance
                        and uy == 0):
                        uy = -1 if ty > ay else 1
                        break

        return ux, uy

