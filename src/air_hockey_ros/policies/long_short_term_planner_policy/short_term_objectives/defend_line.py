from typing import Dict, List, Tuple
from enum import IntEnum
from .objective import Objective

class DefendLine(Objective):
    def __init__(self, agent_id, teammate_ids, commands, rules: Dict, **params):
        super().__init__(agent_id, commands, teammate_ids, rules, **params)
        self.paddle_radius = rules.get('paddle_radius', 20)
        self.y_center = rules.get('height', 600) / 2.0
        self.limit_x = rules.get('width', 800) - self.paddle_radius
        puck_radius = rules.get('puck_radius', 12)
        self.puck_distance = self.paddle_radius + puck_radius
        self.unit_speed_px = rules.get('unit_speed_px', 4.0)
        self.paddle_distance = self.paddle_radius * 2 + self.unit_speed_px
        self.puck_max_speed = rules.get('puck_max_speed', 6.0)
        self.puck_third_speed = self.puck_max_speed / 3.0
        self.rows: List[Tuple[float, float]] = params['defense_rows']
        row = params.get('defense_row', rules.get('goal_offset', 40) + self.paddle_radius + self.unit_speed_px)
        half_unit = self.unit_speed_px / 2
        self.min_x_band = row - half_unit
        self.max_x_band = row + half_unit
        self.min_encounter = self.min_x_band - self.paddle_radius
        self.max_encounter = self.max_x_band + self.paddle_radius
        threat_multiplier = params.get('threat_multiplier', 2)
        self.puck_threat_distance = self.puck_distance * threat_multiplier
        self.lead_coef = params.get('lead_coef', 0.5)

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
        """
        Called when switching into DefendLine after non-offensive objectives.
        Queue size at entry: 0..2 (min-limit). Usually followed by continue_step().
        Params: row (int) -> select band immediately.
        """
        # 1) select row immediately (explicit requirement)
        self.new_ws_step(ws)

    def new_ws_step(self, ws: dict):
        """
        Reactive update when a NEW world state arrives.
        Lightweight, geometry-aware, and limited to ≤2 commands per call.
        """
        # --- basic unpacking ---
        ax = ws["agent_x"][self.agent_id]
        ay = ws["agent_y"][self.agent_id]
        px = ws["puck_x"]
        py = ws["puck_y"]
        self.pvx = ws["puck_vx"]
        self.pvy = ws["puck_vy"]

        adv_x, adv_y = self.commands.get_advance()
        self.exp_x = ax + adv_x * self.unit_speed_px
        self.exp_y = ay + adv_y * self.unit_speed_px

        # --- future puck projection ---
        future_steps = self.commands.get_size() + 1
        self.fut_px = px + self.pvx * future_steps
        self.fut_py = py + self.pvy * future_steps

        # --- quick threat checks ---
        # 1. puck behind paddle → move away to unblock + drift toward y_center
        # 2. puck moving away → maintain position but re-center vertically
        if self.command_puck_consideration():
            self.last_ws = ws
            return

        # --- X correction: move into band only if outside ---
        # --- Y correction ---
        self.correction()

        # --- Build compact burst (≤2) ---
        self.burst_command()
        self.last_ws = ws

    def continue_step(self):
        """
        Called when world state hasn't changed and there's no new status/emergency.
        Uses self.last_ws. Adds 1–2 commands (no capacity logic here).
        """
        ws = self.last_ws

        # --- unpack ---
        self.agent_x = ws["agent_x"]
        self.agent_y = ws["agent_y"]
        ax = self.agent_x[self.agent_id]
        ay = self.agent_y[self.agent_id]
        px = ws["puck_x"]
        py = ws["puck_y"]
        self.pvx = ws["puck_vx"]
        self.pvy = ws["puck_vy"]

        # expected paddle pos after queued cmds
        adv_x, adv_y = self.commands.get_advance()
        self.exp_x = ax + adv_x * self.unit_speed_px
        self.exp_y = ay + adv_y * self.unit_speed_px

        # short-horizon puck projection (when our new cmd takes effect)
        future_steps = self.commands.get_size() + 1
        self.fut_px = px + self.pvx * future_steps
        self.fut_py = py + self.pvy * future_steps

        # --- quick non-threat checks ---
        # A) puck will be behind us → gently clear path + drift toward center
        # B) puck moving away from our goal → softly re-center vertically
        if self.command_puck_consideration():
            return

        # --- X correction into band (only if outside) ---
        # --- Y tracking toward projected puck line (discrete lead) ---
        self.correction()

        self.teammate_consideration()

        # --- build ≤2-step burst ---
        self.burst_command()

    def command_puck_consideration(self) -> bool:
        if (self.exp_x - self.fut_px) > self.puck_distance:
            ux = 1 if self.exp_x < self.max_x_band else 0
            if abs(self.exp_y - self.y_center) <= self.unit_speed_px:
                uy = 0
            else:
                uy = 1 if self.exp_y < self.y_center else -1
            self.commands.push((ux, uy))
            return True

        # Future separation
        if self.pvx <= 0:
            return False

        sep_x = self.fut_px - self.exp_x
        sep_y = self.fut_py - self.exp_y

        if sep_x <= self.puck_distance and abs(sep_y) <= (self.puck_distance + self.unit_speed_px):
            uy = -1 if sep_y >= 0 else 1
            ux = 1 if (self.exp_x + self.unit_speed_px) <= self.limit_x else 0
            self.commands.push((ux, uy))
            return True

        dy = self.y_center - self.exp_y
        if abs(dy) <= self.unit_speed_px:
            uy = 0
        else:
            uy = 1 if dy > 0 else -1
        self.commands.push((0, uy))
        return True

    def correction(self) -> None:
        unit_speed_px = self.unit_speed_px
        if self.exp_x < self.min_x_band:
            self.ux = 1
        elif self.exp_x > self.max_x_band:
            self.ux = -1
        else:
            self.ux = 0

        if self.pvy > self.puck_third_speed:
            lead_step = unit_speed_px
        elif self.pvy < -self.puck_third_speed:
            lead_step = -unit_speed_px
        else:
            lead_step = 0
        tgt_y = self.fut_py + int(self.lead_coef * lead_step)
        self.err_y = tgt_y - self.exp_y

        if abs(self.err_y) <= unit_speed_px:
            self.uy = 0
            return
        if self.err_y > 0:
            self.uy = 1
            return
        self.uy = -1
        return

    def teammate_consideration(self) -> None:
        if self.fut_px - self.exp_x > self.puck_threat_distance:
            return
        teammate_ids = self.teammate_ids
        # if a teammate is very close in y within our band, nudge one step away
        if teammate_ids and self.min_x_band <= self.exp_x <= self.max_x_band:
            for tid in teammate_ids:
                tx, ty = self.agent_x[tid], self.agent_y[tid]
                if ((self.min_encounter < tx < self.max_encounter)
                        and abs(ty - self.exp_y) <= self.paddle_distance):
                    if self.uy == 0:  # if we were holding, nudge away from overlap
                        self.uy = -1 if ty > self.exp_y else 1
                    break

    def burst_command(self) -> None:
        ux, uy = self.ux, self.uy
        err = abs(self.err_y)
        u = self.unit_speed_px

        # 1) Large Y error → strong correction (repeat same vector)
        if err > 2 * u:
            self.commands.push_multiple([(ux, uy), (ux, uy)])
            return

        # 2) X-correction path
        if ux != 0:
            if err <= u:
                # Y nearly aligned → finish X gently
                self.commands.push_multiple([(ux, uy), (ux, 0)])
            else:
                # Y not tiny → stronger X step
                self.commands.push_multiple([(ux, uy), (ux, uy)])
            return

        # 3) In-band normal Y tracking
        if uy != 0:
            if err <= u:
                # one nudge is enough; second would overshoot or jitter
                self.commands.push((0, uy))
            else:
                self.commands.push_multiple([(0, uy), (0, uy)])
            return

        # 4) Hold
        self.commands.push((0, 0))
