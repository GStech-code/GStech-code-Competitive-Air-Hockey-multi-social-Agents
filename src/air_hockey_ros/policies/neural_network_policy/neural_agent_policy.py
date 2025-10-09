# air_hockey_ros/policies/neural_agent_policy.py
from __future__ import annotations
from typing import Tuple, List
import numpy as np
import torch

from air_hockey_ros import AgentPolicy
from .multiagent_paddle_net import MultiAgentPaddleNet

def _to_discrete(dx: float, dy: float, thresh: float) -> Tuple[int, int]:
    def d(v: float) -> int:
        if v >= thresh: return 1
        if v <= -thresh: return -1
        return 0
    return d(dx), d(dy)


class NeuralAgentPolicy(AgentPolicy):
    """
    Role-free neural policy (MLP with per-agent encoder + masked pooling).
    - Always returns discrete actions {-1,0,1} (deadzone threshold).
    - Uses teammate_ids/opponent_ids from constructor (world_state has no team ids).
    - All normalizations & constants are precomputed once.
    """

    def __init__(
        self,
        agent_id: int,
        width: float,
        height: float,
        max_speed: float,            # already adjusted by 1.05 (puck speed norm)
        teammate_ids: List[int],     # fixed list, excludes self
        opponent_ids: List[int],     # fixed list
        deadzone: float = 0.33,
        device: str = "cpu",
    ):
        super().__init__(agent_id)
        # --- constants (computed once) ---
        self.agent_id = int(agent_id)
        self.inv_w = 1.0 / float(width)
        self.inv_h = 1.0 / float(height)
        self.inv_v = 1.0 / float(max_speed)  # no clamping; you said max_speed is already 1.05*true
        self.deadzone = float(deadzone)

        # Pre-store team/opponent indices and masks (fixed size lists)
        self.teammate_ids = list(teammate_ids)
        self.opponent_ids = list(opponent_ids)

        self.T = len(self.teammate_ids)
        self.O = len(self.opponent_ids)

        # Masks are all ones if lists are always filled (fixed size)
        self.opp_mask_const = np.ones((self.O,), dtype=np.float32)

        # Constant teammate/opponent flag values for the 5th feature
        self._tm_flag = 1.0
        self._op_flag = 0.0

        # --- model ---
        self.net = MultiAgentPaddleNet(device_name=device, number_teammates=self.T, number_opponents=self.O)

    # -------------
    # Public API
    # -------------
    def update(self, world_state: dict) -> Tuple[int, int]:
        """
        world_state must include:
          - agent_x: List[float]
          - agent_y: List[float]
          - puck_x: float
          - puck_y: float
          - puck_vx: float
          - puck_vy: float
        """
        # Build structured features using precomputed constants / indices
        self_xy, puck_xyvy, team_feats, opp_feats = self._build_struct(world_state)

        # Forward
        dx_f, dy_f = self.net.get_action_struct(self_xy, puck_xyvy, team_feats, opp_feats)
        if self_xy[0] >= 0.5:
            dx_f = -1

        # Always discrete output
        return _to_discrete(dx_f, dy_f, self.deadzone)

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str, strict: bool = True):
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict):
            for key in ("core", "model_state_dict", "state_dict"):
                if key in sd:
                    sd = sd[key]
                    break
            else:
                sd = list(sd.values())[0]
        self.net.load_state_dict(sd, strict=strict)

    # -------------
    # Features
    # -------------
    def _build_struct(self, ws: dict):
        """
        Returns numpy arrays (no device ops here):
          self_xy:   (2,)
          puck_xyvy: (4,)
          team_feats:(T, 5)
          team_mask: (T,)
          opp_feats: (O, 5)
          opp_mask:  (O,)
        """

        # Normalize positions to [0,1]
        ax = [x * self.inv_w for x in ws["agent_x"]]
        ay = [y * self.inv_h for y in ws["agent_y"]]

        sx = ax[self.agent_id]
        sy = ay[self.agent_id]
        self_xy = np.array([sx, sy], dtype=np.float32)

        # Puck (no clamping)
        px = ws["puck_x"] * self.inv_w
        py = ws["puck_y"] * self.inv_h
        pvx = ws["puck_vx"] * self.inv_v
        pvy = ws["puck_vy"] * self.inv_v
        puck_xyvy = np.array([px, py, pvx, pvy], dtype=np.float32)

        # Pre-allocate
        team_feats = np.zeros((self.T, 5), dtype=np.float32)
        opp_feats  = np.zeros((self.O, 5), dtype=np.float32)

        # Fill teammate features
        for slot, j in enumerate(self.teammate_ids):
            xj = float(ax[j])
            yj = float(ay[j])
            team_feats[slot, 0] = xj - sx           # dx_self
            team_feats[slot, 1] = yj - sy           # dy_self
            team_feats[slot, 2] = xj - px           # dx_puck
            team_feats[slot, 3] = yj - py           # dy_puck
            team_feats[slot, 4] = self._tm_flag     # is_teammate = 1.0

        # Fill opponent features
        for slot, j in enumerate(self.opponent_ids):
            xj = float(ax[j]) * self.inv_w
            yj = float(ay[j]) * self.inv_h
            opp_feats[slot, 0] = xj - sx            # dx_self
            opp_feats[slot, 1] = yj - sy            # dy_self
            opp_feats[slot, 2] = xj - px            # dx_puck
            opp_feats[slot, 3] = yj - py            # dy_puck
            opp_feats[slot, 4] = self._op_flag      # is_teammate = 0.0

        return (
            self_xy,
            puck_xyvy,
            team_feats,
            opp_feats,
        )
