"""
train_ppo_rllib.py

Multi-agent PPO training script for the 2v2 Air Hockey simulation included
with this project.

Dependencies:
  pip install ray[rllib]==2.6.0 gymnasium numpy

Usage:
  python train_ppo_rllib.py

This script defines an RLlib-compatible MultiAgentEnv wrapper around the
provided BaseSimulation. It trains a shared PPO policy for all paddles.

Logging:
 - Uses the standard Python logging module.
 - Writes training progress and environment events to both console and
   a file named `train_ppo_rllib.log`.
 - Checkpoints are saved into `./training/checkpoints/`.

"""

from __future__ import annotations
import math
import os
import logging
from typing import Dict, Any

import numpy as np

# RLlib imports
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Box, MultiDiscrete

from base_simulation import BaseSimulation


# ---------------------- Logging setup ----------------------
logger = logging.getLogger("train_ppo")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler("train_ppo_rllib.log")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

# Ensure checkpoints directory exists
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ---------------------- Environment Wrapper ----------------------
class AirHockeyMultiAgentEnv(MultiAgentEnv):
    """MultiAgentEnv wrapper for the BaseSimulation 2v2 air hockey."""

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.num_agents_team_a = config.get("num_agents_team_a", 2)
        self.num_agents_team_b = config.get("num_agents_team_b", 2)
        assert self.num_agents_team_a == 2 and self.num_agents_team_b == 2, "This wrapper assumes 2v2"

        self.sim = BaseSimulation(view=False)
        self.width, self.height = 800, 600
        self.sim.reset_game(self.num_agents_team_a, self.num_agents_team_b,
                            width=self.width, height=self.height,
                            puck_radius=12, paddle_radius=20,
                            friction_per_tick=0.995, puck_max_speed=8.0)

        self.action_space = MultiDiscrete([3, 3])
        obs_dim = 4 + (self.num_agents_team_a + self.num_agents_team_b) * 2
        high = np.array([max(self.width, self.height)] * obs_dim, dtype=np.float32)
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

        self.max_steps = config.get("max_steps", 2000)
        self.goal_limit = config.get("goal_limit", 10)
        self._step_count = 0

    def _agent_str_to_idx(self, aid: str) -> int:
        if aid.startswith("a_"):
            return int(aid.split("_")[1])
        if aid.startswith("b_"):
            return self.num_agents_team_a + int(aid.split("_")[1])
        raise ValueError("Bad agent id")

    def _make_obs(self) -> Dict[str, np.ndarray]:
        ws = self.sim.get_world_state()
        puck_x = ws["puck_x"]
        puck_y = ws["puck_y"]
        puck_vx = ws.get("puck_vx", 0.0)
        puck_vy = ws.get("puck_vy", 0.0)
        agent_x = ws["agent_x"]
        agent_y = ws["agent_y"]

        base = [puck_x, puck_y, puck_vx, puck_vy]
        obs = np.array(base + sum([[ax, ay] for ax, ay in zip(agent_x, agent_y)], []), dtype=np.float32)

        out = {}
        for i in range(self.num_agents_team_a):
            out[f"a_{i}"] = obs.copy()
        for j in range(self.num_agents_team_b):
            out[f"b_{j}"] = obs.copy()
        return out

    def reset(self, *, seed=None, options=None):
        self.sim.reset_game(self.num_agents_team_a, self.num_agents_team_b,
                            width=self.width, height=self.height)
        self._step_count = 0
        obs = self._make_obs()
        return obs

    def step(self, action_dict: Dict[str, Any]):
        commands = []
        for aid, a in action_dict.items():
            idx = self._agent_str_to_idx(aid)
            vx = int(a[0]) - 1
            vy = int(a[1]) - 1
            commands.append((idx, vx, vy))

        self.sim.apply_commands(commands)
        self._step_count += 1

        obs = self._make_obs()
        scores = self.sim.engine.get_scores()
        team_a_score = scores["team_a_score"]
        team_b_score = scores["team_b_score"]

        if not hasattr(self, "_last_scores"):
            self._last_scores = {"a": 0, "b": 0}
        da = team_a_score - self._last_scores["a"]
        db = team_b_score - self._last_scores["b"]
        self._last_scores["a"] = team_a_score
        self._last_scores["b"] = team_b_score

        rewards = {}
        for i in range(self.num_agents_team_a):
            rewards[f"a_{i}"] = float(da - db) * 0.5
        for j in range(self.num_agents_team_b):
            rewards[f"b_{j}"] = float(db - da) * 0.5

        ws = self.sim.get_world_state()
        puck_x = ws["puck_x"]
        puck_y = ws["puck_y"]
        for i in range(self.num_agents_team_a):
            ax = ws["agent_x"][i]
            ay = ws["agent_y"][i]
            dist = math.hypot(puck_x - ax, puck_y - ay)
            rewards[f"a_{i}"] += -0.001 * dist
        for j in range(self.num_agents_team_b):
            idx = self.num_agents_team_a + j
            ax = ws["agent_x"][idx]
            ay = ws["agent_y"][idx]
            dist = math.hypot(puck_x - ax, puck_y - ay)
            rewards[f"b_{j}"] += -0.001 * dist

        done = (team_a_score >= self.goal_limit) or (team_b_score >= self.goal_limit) or (self._step_count >= self.max_steps)
        dones = {aid: done for aid in list(rewards.keys())}
        dones.update({"__all__": done})

        infos = {aid: {} for aid in rewards.keys()}

        return obs, rewards, dones, infos


# ---------------------- Training script ----------------------

def policy_mapping_fn(agent_id, episode, **kwargs):
    return "shared_policy"


def main():
    ray.init(ignore_reinit_error=True)

    env_config = {"num_agents_team_a": 2, "num_agents_team_b": 2}

    def env_creator(env_config_local):
        return AirHockeyMultiAgentEnv(env_config_local)

    tune.register_env("air_hockey_multi", lambda cfg: env_creator(cfg))

    policies = {
        "shared_policy": (None,
                          AirHockeyMultiAgentEnv().observation_space,
                          AirHockeyMultiAgentEnv().action_space,
                          {}),
    }

    algo = (
        PPOConfig()
        .framework("torch")
        .environment(env="air_hockey_multi", env_config=env_config)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=["shared_policy"])
        .rollouts(num_rollout_workers=1)
        .training(train_batch_size=4000)
        .resources(num_gpus=0)
        .build()
    )

    n_iters = 200
    for i in range(n_iters):
        result = algo.train()
        logger.info(f"Iter {i}: reward_mean={result['episode_reward_mean']}")
        if i % 10 == 0:
            chk = algo.save(checkpoint_dir=CHECKPOINT_DIR)
            logger.info(f"Checkpoint saved: {chk}")

    final_chk = algo.save(checkpoint_dir=CHECKPOINT_DIR)
    logger.info(f"Final checkpoint: {final_chk}")

    ray.shutdown()


if __name__ == "__main__":
    main()
