# air_hockey_ros/policies/multiagent_paddle_net.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerAgentEncoder(nn.Module):
    """
    Shared MLP for each non-self agent slot.
    Input per agent (5 dims):
        [ dx_self, dy_self, dx_puck, dy_puck, is_teammate(0/1) ]
    Output: 16-dim embedding.
    """
    def __init__(self, in_dim: int = 5, hid: int = 32, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, out_dim),
            nn.ReLU(),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, N, 5]
        B, N, D = feats.shape
        x = feats.reshape(B * N, D)
        y = self.net(x)
        return y.reshape(B, N, -1)  # [B, N, 16]


class MultiAgentPaddleNet(nn.Module):
    """
    Role-free multi-agent policy with per-agent encoder + masked mean pooling.

    Structured forward:
        forward_struct(self_xy, puck_xyvy, team_feats, team_mask, opp_feats, opp_mask)

    Shapes:
      self_xy:   [B, 2]           (x_self, y_self)  in [0,1]
      puck_xyvy: [B, 4]           (x_puck, y_puck, vx_puck, vy_puck)  vx,vy normalized
      team_feats:[B, T, 5]        (dx_s,dy_s,dx_p,dy_p,is_tm=1)
      team_mask: [B, T]           1 if slot is real, else 0 (you can pass all-ones if fixed)
      opp_feats: [B, O, 5]        (dx_s,dy_s,dx_p,dy_p,is_tm=0)
      opp_mask:  [B, O]           1 if slot is real, else 0

    Output: tanh in [-1,1]^2  (dx, dy)
    """

    def __init__(
        self,
        number_teammates: int,
        number_opponents: int,
        device: str = "cpu",
        per_agent_dim: int = 5,
        per_agent_embed: int = 32,
        head_hidden1: int = 128,
        head_hidden2: int = 64,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.team_mask=self.to_t(np.ones((number_teammates,), dtype=np.float32))
        self.opp_mask=self.to_t(np.ones((number_opponents,), dtype=np.float32))

        # Shared encoder (ally/opponent use the same encoder)
        self.enc = PerAgentEncoder(in_dim=per_agent_dim,
                                   hid=per_agent_embed,
                                   out_dim=per_agent_embed)

        # Policy head: [self(2) + puck(4) + team(32) + opp(32)] = 70
        head_in = 2 + 4 + per_agent_embed + per_agent_embed
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden1),
            nn.ReLU(),
            nn.Linear(head_hidden1, head_hidden2),
            nn.ReLU(),
            nn.Linear(head_hidden2, 2),
        )

        self._init()
        self.to(self.device)

    def to_t(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _init(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _masked_mean(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        emb:  [B, N, D]
        mask: [B, N]  (float 0/1)
        returns [B, D]
        """
        if emb.size(1) == 0:
            # No slots: return zeros of the right shape
            B, _, D = emb.shape
            return emb.new_zeros(B, D)
        mask = mask.unsqueeze(-1)                 # [B,N,1]
        s = (emb * mask).sum(dim=1)               # [B,D]
        denom = mask.sum(dim=1).clamp_min(1.0)    # [B,1]
        return s / denom

    def forward_struct(
        self,
        self_xy: torch.Tensor,      # [B,2]
        puck_xyvy: torch.Tensor,    # [B,4]
        team_feats: torch.Tensor,   # [B,T,5]
        opp_feats: torch.Tensor,    # [B,O,5]
    ) -> torch.Tensor:
        # Move to device + dtype (no work inside the loop)

        # Encode & pool
        team_emb = self.enc(team_feats)                 # [B,T,16]
        opp_emb  = self.enc(opp_feats)                  # [B,O,16]
        team_vec = self._masked_mean(team_emb, self.team_mask)  # [B,16]
        opp_vec  = self._masked_mean(opp_emb,  self.opp_mask)   # [B,16]

        head_in = torch.cat([self_xy, puck_xyvy, team_vec, opp_vec], dim=1)  # [B,38]
        act = self.head(head_in)
        return torch.tanh(act)  # [-1,1]^2

    @torch.no_grad()
    def get_action_struct(
        self,
        self_xy, puck_xyvy, team_feats, opp_feats
    ) -> Tuple[float, float]:
        """
        Convenience single-sample API (numpy inputs, numpy outputs).
        """

        out = self.forward_struct(
            self.to_t(self_xy), self.to_t(puck_xyvy),
            self.to_t(team_feats), self.to_t(opp_feats)
        )
        v = out.squeeze(0).detach().cpu().numpy()
        return float(v[0]), float(v[1])
