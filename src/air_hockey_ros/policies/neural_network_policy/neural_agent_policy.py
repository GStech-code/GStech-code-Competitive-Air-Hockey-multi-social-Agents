# air_hockey_ros/policies/neural_agent_policy.py
from typing import Tuple, Optional, List
import numpy as np
import torch

# Import colleague’s code
from .low_level_agents import (DefensiveAgent, OffensiveAgent, PassingAgent, NeutralAgent,
                               AdaptivePaddleAgent, PolicyType)
from air_hockey_ros import AgentPolicy


def _to_discrete(u: np.ndarray, thresh: float = 0.33) -> Tuple[int, int]:
    """Map continuous actions in [-1,1] to {-1,0,1} with a deadzone."""
    def d(v):
        if v >= thresh: return 1
        if v <= -thresh: return -1
        return 0
    return int(d(float(u[0]))), int(d(float(u[1])))

class NerualAgentPolicy(AgentPolicy):
    """
    Drop-in for your AgentPolicy:
      - update(world_state) -> (dx, dy) where each in {-1, 0, 1}
    """

    def __init__(
        self,
        agent_id: int,
        width: int,
        height: int,
        max_speed: float,
        teammate_ids: List[int],
        opponent_ids: List[int],
        mode: str = "adaptive",  # 'adaptive' or one of: 'defensive','offensive','passing','neutral'
        device: str = "cpu",
        deterministic: bool = True,
        discrete_output: bool = True,
        deadzone: float = 0.33,
    ):
        super().__init__(agent_id)
        self.device = device
        self.width = width
        self.height = height
        self.teammate_ids = teammate_ids
        self.opponent_ids = opponent_ids
        self.max_speed = max_speed
        self.deterministic = deterministic
        self.discrete_output = discrete_output
        self.deadzone = deadzone

        if mode == "adaptive":
            self.agent = AdaptivePaddleAgent().to(device)
            self._assigned_policy: Optional[PolicyType] = None
        else:
            ctor = {
                "defensive": DefensiveAgent,
                "offensive": OffensiveAgent,
                "passing":   PassingAgent,
                "neutral":   NeutralAgent,
            }[mode]
            self.agent = ctor().to(device)
            self._assigned_policy = None  # not used in single-policy mode

    def set_policy_hint(self, policy: Optional[str]):
        """Optionally steer adaptive selector: 'defensive'/'offensive'/'passing'/'neutral' or None."""
        if policy is None:
            self._assigned_policy = None
        else:
            self._assigned_policy = {
                "defensive": PolicyType.DEFENSIVE,
                "offensive": PolicyType.OFFENSIVE,
                "passing":   PolicyType.PASSING,
                "neutral":   PolicyType.NEUTRAL,
            }[policy]

    def update(self, world_state: dict) -> Tuple[int, int]:
        obs = self.build_observation(world_state, self.agent_id)
        with torch.no_grad():
            if isinstance(self.agent, AdaptivePaddleAgent):
                act = self.agent.get_action(obs, assigned_policy=self._assigned_policy, deterministic=self.deterministic)
            else:
                act = self.agent.get_action(obs, deterministic=self.deterministic)

        if self.discrete_output:
            return _to_discrete(act, self.deadzone)
        # If you want continuous outputs for a future “step_size” pipe:
        return float(act[0]), float(act[1])

    # Checkpoint helpers
    def save(self, path: str):
        torch.save(self.agent.state_dict(), path)

    def load(self, path: str, strict: bool = True):
        sd = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(sd, strict=strict)

    def build_observation(self, world_state: dict) -> np.ndarray:
        """
        Produce the 28-dim observation expected by the low-level agents.
        Layout expected by the model:
          [0:3]   paddle (x, y, speed)
          [3:7]   disc (x, y, vx, vy)
          [7:9]   teammate (x, y)
          [9:13]  opponents (x1, y1, x2, y2)
          [13:17] policy encoding one-hot (def/off/pass/neutral)
          [17:20] game state (3 floats)
          [20:28] communication (8 floats)
        All positions and speeds normalized.
        """
        obs = np.zeros(28, dtype=np.float32)

        # Paddle (self)
        agent_x = [x / self.width for x in world_state['agent_x']]
        agent_y = [y / self.height for y in world_state['agent_y']]
        obs[0] = agent_x[self.agent_id]
        obs[1] = agent_y[self.agent_id]
        obs[3] = world_state["puck_x"] / self.width
        obs[4] = world_state["puck_y"] / self.height
        obs[5] = world_state["puck_vx"] / self.max_speed
        obs[6] = world_state["puck_vy"] / self.max_speed

        for teammate_index, teammate_id in self.teammate_ids:
            obs[7 + teammate_index * 2] = agent_x[teammate_id]
            obs[8 + teammate_index * 2] = agent_y[teammate_id]
        for opponent_index, opponent_id in self.opponent_ids:
            obs[9 + opponent_index * 2] = agent_x[opponent_id]
            obs[10 + opponent_index * 2] = agent_y[opponent_id]

        obs[16] = 1

        return obs