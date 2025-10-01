# src/agents/low_level_agents.py
"""
Low-Level Paddle Control Agents
Individual paddle controllers with specialized policies for defensive, offensive, passing, and neutral behaviors.

This version implements **Option A**: the network is built dynamically based on the
output dimension of `_preprocess_observation`, so any feature engineering done in
preprocessing is automatically reflected in the first Linear layer's in_features.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use absolute import to avoid relative import issues
try:
    from environments.hierarchical_env import PolicyType
except ImportError:
    # Fallback for when running tests directly
    from enum import Enum

    class PolicyType(Enum):
        DEFENSIVE = "defensive"
        OFFENSIVE = "offensive"
        PASSING = "passing"
        NEUTRAL = "neutral"


class BasePaddleAgent(nn.Module):
    """
    Base class for all paddle control agents.

    Key change (Option A): we build the MLP *after* a dummy pass through
    `_preprocess_observation` to discover the true processed feature size.
    """

    def __init__(
        self,
        observation_dim: int = 28,
        action_dim: int = 2,
        hidden_dims: List[int] | tuple[int, ...] = (128, 64, 32),
        activation: str = "relu",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = list(hidden_dims)
        self.device = torch.device(device)

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.ReLU()

        # Build network dynamically from preprocessed size
        self.network = self._build_network_from_preprocess().to(self.device)

        # Policy-specific attributes (to be set by subclasses)
        self.policy_type = PolicyType.NEUTRAL
        self.specialization_weight = 1.0

        self.to(self.device)

    # -------------------------------
    # Network construction & weights
    # -------------------------------
    def _build_network_from_preprocess(self) -> nn.Sequential:
        """Build the MLP based on the dimensionality after preprocessing."""
        with torch.no_grad():
            dummy = torch.zeros(1, self.observation_dim, dtype=torch.float32, device=self.device)
            processed = self._preprocess_observation(dummy)
            if processed.dim() != 2:
                raise ValueError(
                    f"_preprocess_observation must return [B, D]. Got shape {tuple(processed.shape)}"
                )
            in_dim = int(processed.shape[-1])

        layers: List[nn.Module] = []
        prev = in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), self.activation, nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, self.action_dim))  # output layer

        net = nn.Sequential(*layers)
        self._initialize_weights(net)
        return net

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        """Xavier init for all Linear layers inside `module`."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -------------------------------
    # Forward & utilities
    # -------------------------------
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation: Tensor of shape [B, observation_dim] or [observation_dim]
        Returns:
            Action tensor of shape [B, action_dim], tanh-clamped to [-1, 1].
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        obs = observation.to(self.device, dtype=torch.float32)

        processed = self._preprocess_observation(obs)

        # Shape guard to surface clear errors if preprocessing & network mismatch
        try:
            first_linear = next(m for m in self.network.modules() if isinstance(m, nn.Linear))
            expected = first_linear.in_features
        except StopIteration:
            expected = None
        if expected is not None and processed.shape[-1] != expected:
            raise RuntimeError(
                f"Preprocess produced {processed.shape[-1]} features, but network expects {expected}."
            )

        raw_action = self.network(processed)
        action = self._postprocess_action(raw_action, obs)
        return torch.tanh(action)

    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Convenience inference wrapper for numpy inputs (shape [D])."""
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_tensor = self.forward(obs_tensor)
            if not deterministic:
                noise = torch.randn_like(action_tensor) * 0.1
                action_tensor = torch.clamp(action_tensor + noise, -1.0, 1.0)
            return action_tensor.squeeze(0).cpu().numpy()

    # -------------------------------
    # Feature helpers & hooks
    # -------------------------------
    def extract_observation_features(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract semantic slices from the **raw** observation (first 28 dims).
        Expected raw layout:
          [0:3]   paddle (x, y, speed)
          [3:7]   disc (x, y, vx, vy)
          [7:9]   teammate (x, y)
          [9:13]  opponents (x1, y1, x2, y2)
          [13:17] policy encoding one-hot
          [17:20] game state
          [20:28] communication
        """
        x = observation  # [B, D>=28]
        return {
            "paddle_state": x[:, :3],
            "disc_info": x[:, 3:7],
            "teammate_pos": x[:, 7:9],
            "opponents": x[:, 9:13],
            "assigned_policy": x[:, 13:17],
            "game_state": x[:, 17:20],
            "communication": x[:, 20:28],
        }

    # Hooks for subclasses
    def _preprocess_observation(self, observation: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def _postprocess_action(self, raw_action: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Specialized Agents
# -----------------------------------------------------------------------------
class DefensiveAgent(BasePaddleAgent):
    """
    Defensive policy: goal protection, interception, and positioning.
    Adds three engineered features in preprocessing: goal_distance, threat_level, disc_distance.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.policy_type = PolicyType.DEFENSIVE
        self.specialization_weight = 1.5
        # Defensive-specific parameters
        self.goal_protection_radius = 0.3
        self.interception_aggression = 0.8
        self.positioning_weight = 2.0

    def _preprocess_observation(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.extract_observation_features(observation)
        paddle_pos = features["paddle_state"][:, :2]
        disc_info = features["disc_info"]
        disc_pos = disc_info[:, :2]
        disc_vel = disc_info[:, 2:4]

        # Assume own goal at (0.0, 0.5) (left-middle)
        goal_x = torch.zeros_like(paddle_pos[:, :1])
        goal_y = torch.ones_like(paddle_pos[:, :1]) * 0.5
        goal_distance = torch.sqrt((paddle_pos[:, 0:1] - goal_x) ** 2 + (paddle_pos[:, 1:2] - goal_y) ** 2)

        # Threat level: dot(disc_vel, vector to goal) clamped to [0, 1]
        disc_to_goal_x = disc_pos[:, 0:1] - goal_x
        disc_to_goal_y = disc_pos[:, 1:2] - goal_y
        threat_level = disc_vel[:, 0:1] * disc_to_goal_x + disc_vel[:, 1:2] * disc_to_goal_y
        threat_level = torch.clamp(threat_level, 0.0, 1.0)

        # Distance from paddle to disc
        disc_distance = torch.sqrt((paddle_pos[:, 0:1] - disc_pos[:, 0:1]) ** 2 + (paddle_pos[:, 1:2] - disc_pos[:, 1:2]) ** 2)

        return torch.cat([observation, goal_distance, threat_level, disc_distance], dim=1)

    def _postprocess_action(self, raw_action: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        # Slight bias toward holding ground (mild negative x)
        defensive_bias = torch.tensor([[-0.2, 0.0]], device=raw_action.device, dtype=raw_action.dtype)
        return raw_action + 0.3 * defensive_bias.expand_as(raw_action)


class OffensiveAgent(BasePaddleAgent):
    """
    Offensive policy: disc pursuit and shot preparation.
    Adds four engineered features: disc_distance, goal_distance, shot_quality, possession_advantage.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.policy_type = PolicyType.OFFENSIVE
        self.specialization_weight = 1.8
        # Offensive-specific parameters
        self.aggression_factor = 1.2
        self.shooting_threshold = 0.4
        self.disc_pursuit_weight = 2.0

    def _preprocess_observation(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.extract_observation_features(observation)
        paddle_pos = features["paddle_state"][:, :2]
        disc_info = features["disc_info"]
        disc_pos = disc_info[:, :2]
        opponents = features["opponents"]

        # Distances
        disc_distance = torch.sqrt(((paddle_pos - disc_pos) ** 2).sum(dim=1, keepdim=True))

        # Opponent goal at (1.0, 0.5)
        opp_goal_x = torch.ones_like(paddle_pos[:, :1])
        opp_goal_y = torch.ones_like(paddle_pos[:, :1]) * 0.5
        goal_distance = torch.sqrt((paddle_pos[:, 0:1] - opp_goal_x) ** 2 + (paddle_pos[:, 1:2] - opp_goal_y) ** 2)

        # Shooting angle/quality proxy (distance to goal)
        shot_vec_x = opp_goal_x - paddle_pos[:, 0:1]
        shot_vec_y = opp_goal_y - paddle_pos[:, 1:2]
        shot_quality = torch.sqrt(shot_vec_x ** 2 + shot_vec_y ** 2)

        # Possession advantage: are we closer to disc than the nearest opponent?
        opp1_pos = opponents[:, :2]
        opp2_pos = opponents[:, 2:4]
        opp1_disc = torch.sqrt(((opp1_pos - disc_pos) ** 2).sum(dim=1, keepdim=True))
        opp2_disc = torch.sqrt(((opp2_pos - disc_pos) ** 2).sum(dim=1, keepdim=True))
        min_opp_dist = torch.minimum(opp1_disc, opp2_disc)
        possession_advantage = torch.clamp(min_opp_dist - disc_distance, min=0.0)

        return torch.cat([
            observation,
            disc_distance,
            goal_distance,
            shot_quality,
            possession_advantage,
        ], dim=1)

    def _postprocess_action(self, raw_action: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        # Bias towards disc direction scaled by proximity
        features = self.extract_observation_features(observation)
        paddle_pos = features["paddle_state"][:, :2]
        disc_pos = features["disc_info"][:, :2]
        dir_vec = torch.cat([disc_pos[:, 0:1] - paddle_pos[:, 0:1], disc_pos[:, 1:2] - paddle_pos[:, 1:2]], dim=1)
        dist = torch.sqrt(((paddle_pos - disc_pos) ** 2).sum(dim=1, keepdim=True))
        intensity = torch.exp(-3.0 * dist).expand(-1, 2)  # stronger when closer
        modified = raw_action + 0.4 * self.disc_pursuit_weight * dir_vec * intensity
        return modified * self.aggression_factor


class PassingAgent(BasePaddleAgent):
    """
    Passing/coordination policy. For now, preprocessing keeps the original 28-dim input
    to keep the model simple and compatible with tests; coordination tweaks happen in postprocess.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.policy_type = PolicyType.PASSING
        self.specialization_weight = 1.3
        self.coordination_weight = 1.5
        self.teammate_awareness = 2.0
        self.optimal_teammate_distance = 0.25

    def _preprocess_observation(self, observation: torch.Tensor) -> torch.Tensor:
        return observation

    def _postprocess_action(self, raw_action: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        # Mild exploratory coordination bias
        coord_bias = torch.randn_like(raw_action) * 0.05
        return raw_action + coord_bias


class NeutralAgent(BasePaddleAgent):
    """Neutral/baseline policy with no specialization."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.policy_type = PolicyType.NEUTRAL
        self.specialization_weight = 1.0

    def _preprocess_observation(self, observation: torch.Tensor) -> torch.Tensor:
        return observation
    def _postprocess_action(self, raw_action: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        return raw_action


# -----------------------------------------------------------------------------
# Factory & Adaptive Agent
# -----------------------------------------------------------------------------
class PaddleAgentFactory:
    """Factory for creating paddle agents of different types."""

    @staticmethod
    def create_agent(policy_type: PolicyType, **kwargs) -> BasePaddleAgent:
        if policy_type == PolicyType.DEFENSIVE:
            return DefensiveAgent(**kwargs)
        elif policy_type == PolicyType.OFFENSIVE:
            return OffensiveAgent(**kwargs)
        elif policy_type == PolicyType.PASSING:
            return PassingAgent(**kwargs)
        elif policy_type == PolicyType.NEUTRAL:
            return NeutralAgent(**kwargs)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

    @staticmethod
    def create_multi_policy_agent(policy_types: List[PolicyType], **kwargs) -> Dict[PolicyType, BasePaddleAgent]:
        return {pt: PaddleAgentFactory.create_agent(pt, **kwargs) for pt in policy_types}


class AdaptivePaddleAgent(nn.Module):
    """
    Adaptive agent that can switch between specialized policies.
    The policy selector operates on the **raw** 28-dim observation (not preprocessed),
    while each specialized policy uses its own preprocessing pipeline.
    """

    def __init__(self, policy_types: Optional[List[PolicyType]] = None, **kwargs) -> None:
        super().__init__()
        if policy_types is None:
            policy_types = [PolicyType.DEFENSIVE, PolicyType.OFFENSIVE, PolicyType.PASSING, PolicyType.NEUTRAL]

        # Create specialized agents for each policy
        self.policy_agents = PaddleAgentFactory.create_multi_policy_agent(policy_types, **kwargs)

        # Lightweight policy selection head on raw 28-dim observation
        self.policy_selector = nn.Sequential(
            nn.Linear(28, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(policy_types)),
        )

        self.current_policy = PolicyType.NEUTRAL
        self.policy_types = policy_types

    def forward(self, observation: torch.Tensor, assigned_policy: Optional[PolicyType] = None) -> torch.Tensor:
        if assigned_policy is not None:
            return self.policy_agentsassigned_policy

        # Learned selection on raw obs
        policy_logits = self.policy_selector(observation)
        policy_probs = F.softmax(policy_logits, dim=1)
        selected_policy_idx = torch.argmax(policy_probs, dim=1)

        # Route each item in the batch to its selected policy
        actions = []
        for i, obs in enumerate(observation):
            policy_idx = selected_policy_idx[i].item()
            selected_policy = self.policy_types[policy_idx]
            policy_agent = self.policy_agents[selected_policy]
            action = policy_agent(obs.unsqueeze(0))
            actions.append(action)
        return torch.cat(actions, dim=0)

    @torch.no_grad()
    def get_action(self, observation: np.ndarray, assigned_policy: Optional[PolicyType] = None, deterministic: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        if assigned_policy is not None:
            return self.policy_agents[assigned_policy].get_action(observation, deterministic)
        action = self.forward(obs_tensor)
        if not deterministic:
            noise = torch.randn_like(action) * 0.1
            action = torch.clamp(action + noise, -1.0, 1.0)
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def get_policy_weights(self, observation: np.ndarray) -> Dict[PolicyType, float]:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        probs = F.softmax(self.policy_selector(obs_tensor), dim=1)[0]
        return {pt: probs[i].item() for i, pt in enumerate(self.policy_types)}