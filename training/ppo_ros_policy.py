"""
Standalone PPO policy classes for ROS integration.

These classes have no external dependencies and can be safely pickled/unpickled.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import numpy as np


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


class PPOAgentPolicy:
    """
    ROS-compatible wrapper for trained PPO policy.
    
    This class is fully self-contained and doesn't require external modules
    to be unpickled. The network is stored as a state dict and rebuilt on load.
    """
    
    def __init__(self, 
                 network_state_dict: Dict,
                 obs_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 agent_id: int,
                 num_team_agents: int,
                 num_opponent_agents: int,
                 scenario_params: Dict,
                 device: str = "cpu"):
        """Initialize PPO agent policy"""
        self.agent_id = agent_id
        
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
        self.num_team_agents = num_team_agents
        self.num_opponent_agents = num_opponent_agents
        self.num_total_agents = num_team_agents + num_opponent_agents
        
        # Normalization constants
        self.inv_w = 1.0 / float(scenario_params.get('width', 800))
        self.inv_h = 1.0 / float(scenario_params.get('height', 600))
        puck_max = float(scenario_params.get('puck_max_speed', 6.0))
        unit_speed = float(scenario_params.get('unit_speed_px', 4.0))
        self.inv_v = 1.0 / (max(puck_max, unit_speed) * 1.05)
    
    def update(self, world_state: Dict) -> Tuple[int, int]:
        """Get action for current world state (called by ROS agent node)"""
        # Build observation from world state
        obs = self._build_observation(world_state)
        
        # Run inference
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_mean, _ = self.network(obs_tensor)
            action = action_mean.squeeze(0).cpu().numpy()
        
        # Discretize to {-1, 0, 1}
        vx = self._discretize(action[0])
        vy = self._discretize(action[1])
        
        return vx, vy
    
    def on_agent_init(self):
        """Called when agent is initialized (ROS compatibility)"""
        pass
    
    def on_agent_close(self):
        """Called when agent is closed (ROS compatibility)"""
        pass
    
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
        
        # Self position (2 dims)
        obs[idx:idx+2] = [ax[self.agent_id], ay[self.agent_id]]
        idx += 2
        
        # Puck state (4 dims)
        obs[idx:idx+4] = [px, py, pvx, pvy]
        idx += 4
        
        # Teammate features (exclude self)
        for j in range(self.num_team_agents):
            if j != self.agent_id:
                obs[idx:idx+5] = [
                    ax[j] - ax[self.agent_id],
                    ay[j] - ay[self.agent_id],
                    ax[j] - px,
                    ay[j] - py,
                    1.0
                ]
                idx += 5
        
        # Opponent features
        for j in range(self.num_team_agents, self.num_total_agents):
            obs[idx:idx+5] = [
                ax[j] - ax[self.agent_id],
                ay[j] - ay[self.agent_id],
                ax[j] - px,
                ay[j] - py,
                0.0
            ]
            idx += 5
        
        return obs
    
    def _discretize(self, val: float) -> int:
        """Convert continuous action to discrete {-1, 0, 1}"""
        if val > 0.33:
            return 1
        elif val < -0.33:
            return -1
        return 0