# air_hockey_ros/policies/ppo_wrappers.py
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

# Do NOT import your agent/team policies here; we are not modifying them.
# We only depend on MultiAgentPaddleNet as the shared encoder.
from .multiagent_paddle_net import MultiAgentPaddleNet


# ──────────────────────────────────────────────────────────────────────────────
# Factorized (X,Y) head: each axis has 3 logits (→ {-1,0,1}); value is scalar.
# This wrapper is a drop-in AgentPolicy *container* (not modifying your originals).
# It computes its own action (and PPO stats) from world_state and returns (dx,dy).
# ──────────────────────────────────────────────────────────────────────────────

_AXIS_TO_INT = {-1: 0, 0: 1, 1: 2}
_INT_TO_AXIS = {0: -1, 1: 0, 2: 1}

class _AxisHead(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, out_dim),
        )
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class _ValueHead(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, 1),
        )
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B,1]


class AgentPPOWrapper:
    """
    Container that behaves like your AgentPolicy:
      update(world_state) -> (dx, dy) in {-1,0,1}

    It owns:
      - shared MultiAgentPaddleNet encoder (passed-in reference)
      - per-agent heads: logits_x (3), logits_y (3), value (1)
    It caches (feat, ax, ay, logp, value) for PPO.

    """
    def __init__(
        self,
        agent_id: int,
        width: float,
        height: float,
        max_speed: float,
        teammate_ids: List[int],
        opponent_ids: List[int],
        shared_core: MultiAgentPaddleNet,
        device: str = "cpu",
    ):
        self.agent_id = int(agent_id)
        self.half_line = width / 2.0
        self.inv_w = 1.0 / float(width)
        self.inv_h = 1.0 / float(height)
        self.inv_v = 1.0 / float(max_speed)
        self.device = torch.device(device)
        self.T = len(teammate_ids)
        self.O = len(opponent_ids)
        self.teammate_ids = teammate_ids
        self.opponent_ids = opponent_ids

        self.core = shared_core  # shared encoder (already on device)
        feat_dim = 2 + 4 + self.core.per_agent_embed + self.core.per_agent_embed

        self.head_x = _AxisHead(feat_dim, hid=64, out_dim=3).to(self.device)
        self.head_y = _AxisHead(feat_dim, hid=64, out_dim=3).to(self.device)
        self.v_head = _ValueHead(feat_dim, hid=64).to(self.device)

        self.training_mode = False

        # PPO caches (torch tensors on device unless stated)
        self.last_feat: Optional[torch.Tensor] = None   # [1, F]
        self.last_ax: Optional[int] = None              # int in {0,1,2}
        self.last_ay: Optional[int] = None
        self.last_logp: Optional[torch.Tensor] = None   # scalar
        self.last_value: Optional[torch.Tensor] = None  # scalar

    # ----- Public API: identical shape to your AgentPolicy -----
    @torch.no_grad()
    def update(self, ws: Dict) -> Tuple[int, int]:
        feat = self._build_feature(ws)                                  # [1, F]
        logits_x, logits_y = self.head_x(feat), self.head_y(feat)       # [1,3], [1,3]
        value = self.v_head(feat).squeeze(-1)                            # [1]

        dist_x = torch.distributions.Categorical(logits=logits_x)
        dist_y = torch.distributions.Categorical(logits=logits_y)

        if self.training_mode:
            ax = int(dist_x.sample().item())
            ay = int(dist_y.sample().item())
        else:
            ax = int(torch.argmax(logits_x, dim=-1).item())
            ay = int(torch.argmax(logits_y, dim=-1).item())

        # Decode to {-1, 0, 1}
        dx = _INT_TO_AXIS[ax]
        dy = _INT_TO_AXIS[ay]

        # ----- HALF-LINE ENFORCEMENT (world units) -----
        # Force agent to move left (-1) if they're at or past the half line
        if ws["agent_x"][self.agent_id] >= self.half_line:
            if dx != -1:
                dx = -1  # Force move left
                ax = _AXIS_TO_INT[dx]  # Convert back to action index for logging

        # Compute log-prob AFTER enforcement (must match the actual executed action)
        logp = dist_x.log_prob(torch.tensor(ax, device=self.device)) \
            + dist_y.log_prob(torch.tensor(ay, device=self.device))

        # Cache AFTER all adjustments so PPO uses the executed action/values
        self.last_feat = feat.detach()
        self.last_ax = ax
        self.last_ay = ay
        self.last_logp = logp.detach()
        self.last_value = value.detach()

        # Apply mirroring AFTER caching (for Team B's world view)
        if self.mirror_xy:
            dx = -dx  # mirror X only (world is mirrored for team B)
        
        return dx, dy

    def set_train_mode(self, on: bool):
        self.training_mode = bool(on)
        if on:
            self.core.train()
            self.head_x.train()
            self.head_y.train()
            self.v_head.train()
        else:
            self.core.eval()
            self.head_x.eval()
            self.head_y.eval()
            self.v_head.eval()

    # ----- Params for optimizer -----
    def parameters(self):
        # note: encoder is shared; trainer collects it once at team level
        for p in self.head_x.parameters(): yield p
        for p in self.head_y.parameters(): yield p
        for p in self.v_head.parameters(): yield p

    # ----- Feature builder (consistent with your encoder) -----
    def _build_feature(self, ws: Dict) -> torch.Tensor:
        # Normalize coordinates/speeds
        ax = [x * self.inv_w for x in ws["agent_x"]]
        ay = [y * self.inv_h for y in ws["agent_y"]]

        px = ws["puck_x"] * self.inv_w
        py = ws["puck_y"] * self.inv_h
        pvx = ws["puck_vx"] * self.inv_v
        pvy = ws["puck_vy"] * self.inv_v

        sx = ax[self.agent_id]; sy = ay[self.agent_id]
        self_xy = np.array([sx, sy], dtype=np.float32)
        puck_xyvy = np.array([px, py, pvx, pvy], dtype=np.float32)

        team_feats = np.zeros((self.T, 5), dtype=np.float32)
        opp_feats = np.zeros((self.O, 5), dtype=np.float32)

        for slot, j in enumerate(self.teammate_ids):
            xj, yj = float(ax[j]), float(ay[j])
            team_feats[slot, 0] = xj - sx
            team_feats[slot, 1] = yj - sy
            team_feats[slot, 2] = xj - px
            team_feats[slot, 3] = yj - py
            team_feats[slot, 4] = 1.0

        for slot, j in enumerate(self.opponent_ids):
            xj, yj = float(ax[j]), float(ay[j])
            opp_feats[slot, 0] = xj - sx
            opp_feats[slot, 1] = yj - sy
            opp_feats[slot, 2] = xj - px
            opp_feats[slot, 3] = yj - py
            opp_feats[slot, 4] = 0.0

        # Encode team/opponents with shared core encoder + masked means
        self_xy_t = torch.as_tensor(self_xy, device=self.device).unsqueeze(0)
        puck_xyvy_t = torch.as_tensor(puck_xyvy, device=self.device).unsqueeze(0)

        chunks = [self_xy_t, puck_xyvy_t]

        if self.T > 0:
            t_feats = torch.as_tensor(team_feats, device=self.device).unsqueeze(0)  # [1,T,5]
            t_emb = self.core.enc(t_feats)
            t_vec = self.core._masked_mean(t_emb, self.core.team_mask)              # [1,E]
            chunks.append(t_vec)
        else:
            chunks.append(torch.zeros(1, self.core.per_agent_embed, device=self.device))

        if self.O > 0:
            o_feats = torch.as_tensor(opp_feats, device=self.device).unsqueeze(0)   # [1,O,5]
            o_emb = self.core.enc(o_feats)
            o_vec = self.core._masked_mean(o_emb, self.core.opp_mask)               # [1,E]
            chunks.append(o_vec)
        else:
            chunks.append(torch.zeros(1, self.core.per_agent_embed, device=self.device))

        feat = torch.cat(chunks, dim=1).float()  # [1, F]
        return feat

    def on_agent_close(self):
        # Mirrors the original neural agent policy's no-op close hook.
        # Useful placeholder if you later add threads, buffers, or file handles.
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Team wrapper: owns ONE shared encoder; builds AgentPPOWrapper per agent.
# We keep it very small; trainer can create/destroy these per episode.
# ──────────────────────────────────────────────────────────────────────────────

class TeamPPOWrapper:
    def __init__(
        self,
        agent_ids: List[int],
        width: float, height: float, max_speed: float,
        teammates_lists: List[List[int]],
        opponents_list: List[int],
        device: str = "cpu",
    ):
        self.device = device
        self.core = MultiAgentPaddleNet(
            number_teammates=0, number_opponents=0, device_name=device
        )
        self.reset(agent_ids, width, height, max_speed, teammates_lists, opponents_list)

    def reset(self, agent_ids, width, height, max_speed, teammates_lists, opponents_list):
        self.agent_ids = agent_ids
        T = len(teammates_lists[0]) if teammates_lists else 0
        O = len(opponents_list) if opponents_list else 0
        self.core.reset_num_agents(T, O)
        self.policies: List[AgentPPOWrapper] = []
        for agent_id, tmates in zip(agent_ids, teammates_lists):
            pol = AgentPPOWrapper(
                agent_id=agent_id,
                width=width, height=height, max_speed=max_speed,
                teammate_ids=tmates, opponent_ids=opponents_list,
                shared_core=self.core, device=self.device
            )
            self.policies.append(pol)

    def parameters(self):
        # shared encoder + all heads
        for p in self.core.parameters():
            yield p
        for pol in self.policies:
            for p in pol.parameters():
                yield p

    def set_train_mode(self, on: bool):
        self.core.train(mode=on)
        for p in self.policies:
            p.set_train_mode(on)
