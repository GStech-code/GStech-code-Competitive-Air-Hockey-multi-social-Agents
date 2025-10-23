# This file is originaly from training/ppo_ros_policy.py, copied for sake of comfortability.
"""
Standalone PPO policy classes for ROS integration.

These classes have no external dependencies and can be safely pickled/unpickled.
"""

from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import numpy as np
from air_hockey_ros import AgentPolicy

class ActorCriticStandalone(nn.Module):
    """Standalone Actor-Critic network (no external dependencies)"""
    
    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution and value"""
        features = self.shared_net(obs)
        action_mean = torch.tanh(self.actor_mean(features))
        value = self.critic(features)
        return action_mean, value


class PPOAgentPolicy(AgentPolicy):
    """
    ROS-compatible wrapper for trained PPO policy.
    
    This class is fully self-contained and doesn't require external modules
    to be unpickled. The network is stored as a state dict and rebuilt on load.
    """
    
    def __init__(self,
                 agent_id: int,
                 teammate_ids: List[int],
                 opponent_ids: List[int],
                 cap_agents_per_team: int,
                 device: str,
                 network_state_dict: Dict,
                 obs_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 width: float,
                 height: float,
                 half_line_distance: float,
                 max_speed: float):
        """Initialize PPO agent policy"""
        super().__init__(agent_id)
        
        # Store network config
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # Build network from state dict
        self.network = ActorCriticStandalone(obs_dim, hidden_dim, action_dim)
        self.network.load_state_dict(network_state_dict)
        self.device = torch.device(device)
        self.network.to(self.device)
        self.network.eval()
        
        # Team configuration
        self.teammate_ids = teammate_ids[:cap_agents_per_team - 1]
        self.opponent_ids = opponent_ids[:cap_agents_per_team]
        self.cap_agents_per_team = cap_agents_per_team

        self.zero_teammates = []
        for _ in range(cap_agents_per_team - 1 - len(self.teammate_ids)):
            self.zero_teammates.extend([0, 0, 0, 0, 1])

        self.zero_opponents = []
        for _ in range(cap_agents_per_team - len(self.opponent_ids)):
            self.zero_opponents.extend([1, 1, 1, 1, 0])

        
        # Normalization constants
        # float(params.get('width', 800))
        # float(params.get('height', 600))
        # puck_max = float(params.get('puck_max_speed', 6.0))
        # unit_speed = float(params.get('unit_speed_px', 4.0))
        # (max(puck_max, unit_speed) * 1.05)
        self.inv_w = 1.0 / width
        self.no_cross_points = ((width / 2) - half_line_distance) * self.inv_w
        self.inv_h = 1.0 / height

        self.inv_v = 1.0 / max_speed
    
    def update(self, world_state: Dict) -> Tuple[int, int]:
        """Get action for current world state (called by ROS agent node)"""
        # Build observation from world state
        obs, current_x = self._build_observation(world_state)
        
        # Run inference
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_mean, _ = self.network(obs_tensor)
            action = action_mean.squeeze(0).cpu().numpy()
        
        # Discretize to {-1, 0, 1}
        vx = self._discretize(action[0])
        if vx == 1:
            if current_x == self.no_cross_points:
                vx = 0
            elif current_x > self.no_cross_points:
                vx = -1
        vy = self._discretize(action[1])
        
        return vx, vy
    
    def _build_observation(self, world_state: Dict) -> np.ndarray:
        """Build observation vector from world state"""
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        # Normalize positions and velocities
        ax = [x * self.inv_w for x in world_state["agent_x"]]
        ay = [y * self.inv_h for y in world_state["agent_y"]]
        px = world_state["puck_x"] * self.inv_w
        py = world_state["puck_y"] * self.inv_h
        pvx = world_state["puck_vx"] * self.inv_v
        pvy = world_state["puck_vy"] * self.inv_v
        
        idx = 0
        aix = ax[self.agent_id]
        aiy = ay[self.agent_id]
        
        # Self position (2 dims)
        obs[idx:idx+2] = [aix, aiy]
        idx += 2
        
        # Puck state (4 dims)
        obs[idx:idx+4] = [px, py, pvx, pvy]
        idx += 4
        
        # Teammate features (exclude self)
        for t in self.teammate_ids:
            obs[idx:idx + 5] = [
                ax[t] - aix,
                ay[t] - aiy,
                ax[t] - px,
                ay[t] - py,
                1.0
            ]
            idx += 5

        for val in self.zero_teammates:
            obs[idx] = val
            idx += 1

        # Opponent features
        for o in self.opponent_ids:
            obs[idx:idx + 5] = [
                ax[o] - aix,
                ay[o] - aiy,
                ax[o] - px,
                ay[o] - py,
                0.0
            ]
            idx += 5

        for val in self.zero_opponents:
            obs[idx] = val
            idx += 1
        
        return obs, aix
    
    def _discretize(self, val: float) -> int:
        """Convert continuous action to discrete {-1, 0, 1}"""
        if val > 0.33:
            return 1
        elif val < -0.33:
            return -1
        return 0