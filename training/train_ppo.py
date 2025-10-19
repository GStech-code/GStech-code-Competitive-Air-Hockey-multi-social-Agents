"""
PPO Training for Multi-Agent Air Hockey - FIXED VERSION

Key improvements:
1. Dense rewards for approaching puck (Phase 1 now works!)
2. Proximity rewards for being near puck
3. Proper contact detection (unchanged)
4. All reward ratios preserved

Usage:
    python train_ppo.py --config config/ppo_config.yaml
"""

from __future__ import annotations
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time
import logging
import argparse

# Import from the project
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from air_hockey_ros.simulations.base import BaseEngine
from air_hockey_ros.policies.neural_network_policy.multiagent_paddle_net import MultiAgentPaddleNet


@dataclass
class PPOConfig:
    """PPO hyperparameters"""
    # Training
    total_timesteps: int = 1_000_000
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    
    # Environment
    n_envs: int = 1
    num_agents_team_a: int = 2
    num_agents_team_b: int = 2
    
    # Network
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim: int = 128
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


class AirHockeyEnv:
    """Gym-like wrapper for BaseEngine simulation"""
    
    def __init__(self, config: PPOConfig, scenario_params: Dict):
        self.config = config
        self.scenario_params = scenario_params
        self.engine = BaseEngine()
        self.engine.configure(**scenario_params)
        
        # Team configuration
        self.num_team_a = config.num_agents_team_a
        self.num_team_b = config.num_agents_team_b
        self.num_agents = self.num_team_a + self.num_team_b
        
        # Observation/action space info
        self.obs_dim = self._get_obs_dim()
        self.action_dim = 2  # (vx, vy)
        
        # Episode limits (can be overridden by curriculum)
        self.max_score = 3
        self.max_steps = 1200
        
        # Normalization constants
        self.inv_w = 1.0 / float(scenario_params.get('width', 800))
        self.inv_h = 1.0 / float(scenario_params.get('height', 600))
        puck_max = float(scenario_params.get('puck_max_speed', 6.0))
        unit_speed = float(scenario_params.get('unit_speed_px', 4.0))
        self.inv_v = 1.0 / (max(puck_max, unit_speed) * 1.05)
    
    def _get_obs_dim(self) -> int:
        """Calculate observation dimension"""
        # Self position (2) + puck state (4) + teammate features + opponent features
        obs_per_agent = 2 + 4 + (self.num_team_a - 1) * 5 + self.num_team_b * 5
        return obs_per_agent
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.engine.reset(self.num_team_a, self.num_team_b)
        return self._get_obs()
    
    def step(self, actions: np.ndarray, reward_weights: Dict = None) -> Tuple:
        """
        Step the environment.
        
        Args:
            actions: Action array [num_agents, 2]
            reward_weights: Optional reward weights from curriculum config
        """
        # Store previous state
        prev_state = self.engine.get_world_state()
        
        # Convert continuous actions to discrete commands
        commands = []
        for i, action in enumerate(actions):
            vx = self._discretize(action[0])
            vy = self._discretize(action[1])
            commands.append((i, vx, vy))
        
        # Execute actions
        self.engine.apply_commands(commands)
        self.engine.step()
        
        # Get new state
        new_state = self.engine.get_world_state()
        
        # Compute rewards
        rewards = self._compute_rewards(prev_state, new_state, actions, 
                                        reward_weights=reward_weights)
        
        # Check done
        done = self._check_done(new_state)
        
        # Get observations
        obs = self._get_obs()
        
        # Info
        info = {
            'scores': {
                'team_a_score': new_state['team_a_score'],
                'team_b_score': new_state['team_b_score']
            }
        }
        
        return obs, rewards, done, info
    
    def _discretize(self, val: float) -> int:
        """Convert continuous action [-1, 1] to discrete {-1, 0, 1}"""
        if val > 0.33:
            return 1
        elif val < -0.33:
            return -1
        return 0
    
    def _get_obs(self) -> np.ndarray:
        """Get observations for team A agents"""
        ws = self.engine.get_world_state()
        obs = np.zeros((self.num_team_a, self.obs_dim), dtype=np.float32)
        
        # Normalize positions
        ax = [x * self.inv_w for x in ws["agent_x"]]
        ay = [y * self.inv_h for y in ws["agent_y"]]
        px = ws["puck_x"] * self.inv_w
        py = ws["puck_y"] * self.inv_h
        pvx = ws["puck_vx"] * self.inv_v
        pvy = ws["puck_vy"] * self.inv_v
        
        for i in range(self.num_team_a):
            idx = 0
            # Self position
            obs[i, idx:idx+2] = [ax[i], ay[i]]
            idx += 2
            
            # Puck state
            obs[i, idx:idx+4] = [px, py, pvx, pvy]
            idx += 4
            
            # Teammate features (exclude self)
            for j in range(self.num_team_a):
                if j != i:
                    obs[i, idx:idx+5] = [
                        ax[j] - ax[i],
                        ay[j] - ay[i],
                        ax[j] - px,
                        ay[j] - py,
                        1.0  # is_teammate
                    ]
                    idx += 5
            
            # Opponent features
            for j in range(self.num_team_a, self.num_agents):
                obs[i, idx:idx+5] = [
                    ax[j] - ax[i],
                    ay[j] - ay[i],
                    ax[j] - px,
                    ay[j] - py,
                    0.0  # is_opponent
                ]
                idx += 5
        
        return obs
    
    def _compute_rewards(self, prev_state: Dict, new_state: Dict, actions: np.ndarray,
                        reward_weights: Dict = None) -> np.ndarray:  # ← ADD THIS PARAMETER
        """
        Compute rewards using configurable weights.
        
        Args:
            prev_state: Previous world state
            new_state: Current world state
            actions: Agent actions
            reward_weights: Dictionary of reward weights from config (optional)
        
        Returns:
            Reward array for each agent
        """
        # Default weights if none provided (for backward compatibility)
        if reward_weights is None:
            reward_weights = {
                'goal_scored': 1000.0,
                'goal_conceded': -1000.0,
                'approach_puck': 5.0,
                'proximity_to_puck': 20.0,
                'contact_puck': 100.0,
                'shoot_toward_goal': 200.0,
                'defensive_position': 0.0,
                'center_coverage': 0.0,
                'teammate_collision': 0.0,
                'action_penalty': 0.0,
                'unnecessary_movement': 0.0,
            }
        
        rewards = np.zeros(self.num_team_a, dtype=np.float32)
        
        # === 1. GOAL REWARDS ===
        if new_state['team_a_score'] > prev_state['team_a_score']:
            rewards += reward_weights.get('goal_scored', 1000.0)
        if new_state['team_b_score'] > prev_state['team_b_score']:
            rewards += reward_weights.get('goal_conceded', -1000.0)
        
        # Get positions
        puck_x = new_state['puck_x']
        puck_y = new_state['puck_y']
        prev_puck_x = prev_state['puck_x']
        prev_puck_y = prev_state['puck_y']
        half_line = self.engine.width / 2.0
        goal_center_y = self.engine.height / 2.0
        paddle_radius = self.engine.paddle_radius
        puck_radius = self.engine.puck_radius
        hit_distance = paddle_radius + puck_radius + 15
        
        # Check if puck in defensive zone (for conditional rewards)
        puck_in_defensive_zone = (puck_x < half_line)
        
        # === 2. APPROACH REWARDS ===
        w_approach = reward_weights.get('approach_puck', 5.0)
        for i in range(self.num_team_a):
            agent_x = new_state['agent_x'][i]
            agent_y = new_state['agent_y'][i]
            prev_agent_x = prev_state['agent_x'][i]
            prev_agent_y = prev_state['agent_y'][i]
            
            dist_to_puck = np.hypot(agent_x - puck_x, agent_y - puck_y)
            prev_dist_to_puck = np.hypot(prev_agent_x - prev_puck_x, prev_agent_y - prev_puck_y)
            
            # Approach reward
            if w_approach > 0:
                distance_improvement = prev_dist_to_puck - dist_to_puck
                if distance_improvement > 0:
                    approach_reward = w_approach * np.tanh(distance_improvement / 20.0)
                    rewards[i] += approach_reward
            
            # === 3. PROXIMITY REWARD ===
            w_proximity = reward_weights.get('proximity_to_puck', 20.0)
            if w_proximity > 0 and dist_to_puck < hit_distance * 2.0:
                if dist_to_puck < hit_distance:
                    push_factor = (hit_distance - dist_to_puck) / hit_distance
                    push_reward = w_proximity * push_factor
                    rewards[i] += push_reward
                else:
                    range_factor = (hit_distance * 2.0 - dist_to_puck) / hit_distance
                    approach_bonus = (w_proximity * 0.25) * range_factor
                    rewards[i] += approach_bonus
        
        # === 4. CONTACT DETECTION ===
        new_puck_vx = new_state['puck_vx']
        new_puck_vy = new_state['puck_vy']
        prev_puck_vx = prev_state.get('puck_vx', 0.0)
        prev_puck_vy = prev_state.get('puck_vy', 0.0)
        
        delta_puck_vx = new_puck_vx - prev_puck_vx
        delta_puck_vy = new_puck_vy - prev_puck_vy
        puck_vel_change = np.hypot(delta_puck_vx, delta_puck_vy)
        
        w_contact = reward_weights.get('contact_puck', 100.0)
        w_shooting = reward_weights.get('shoot_toward_goal', 200.0)
        
        if puck_vel_change > 0.3:
            best_agent = None
            best_alignment = -1.0
            
            for i in range(self.num_team_a):
                prev_agent_x = prev_state['agent_x'][i]
                prev_agent_y = prev_state['agent_y'][i]
                agent_x = new_state['agent_x'][i]
                agent_y = new_state['agent_y'][i]
                
                dist_to_puck_now = np.hypot(agent_x - puck_x, agent_y - puck_y)
                
                if dist_to_puck_now < hit_distance * 1.2:
                    agent_dx = agent_x - prev_agent_x
                    agent_dy = agent_y - prev_agent_y
                    agent_movement = np.hypot(agent_dx, agent_dy)
                    
                    if agent_movement > 0.05:
                        agent_dx_norm = agent_dx / agent_movement
                        agent_dy_norm = agent_dy / agent_movement
                        delta_vx_norm = delta_puck_vx / puck_vel_change
                        delta_vy_norm = delta_puck_vy / puck_vel_change
                        
                        alignment = agent_dx_norm * delta_vx_norm + agent_dy_norm * delta_vy_norm
                        
                        if alignment > best_alignment:
                            best_alignment = alignment
                            best_agent = i
            
            # === 5. CONTACT REWARD ===
            if best_agent is not None and best_alignment > 0.2 and w_contact > 0:
                rewards[best_agent] += w_contact
                
                # === 6. SHOOTING REWARD ===
                if new_puck_vx > 0 and w_shooting > 0:  # Moving toward opponent goal
                    shot_power = min(new_puck_vx / self.engine.puck_max_speed, 1.0)
                    rewards[best_agent] += w_shooting * shot_power
        
        # === 7. DEFENSIVE REWARDS (conditional on puck location) ===
        w_defense = reward_weights.get('defensive_position', 0.0)
        w_coverage = reward_weights.get('center_coverage', 0.0)
        
        if puck_in_defensive_zone and (w_defense > 0 or w_coverage > 0):
            for i in range(self.num_team_a):
                agent_x = new_state['agent_x'][i]
                agent_y = new_state['agent_y'][i]
                
                # Defensive positioning
                if w_defense > 0 and agent_x < puck_x:  # Between puck and own goal
                    ideal_x = puck_x * 0.3
                    ideal_y = puck_y
                    dist_from_ideal = np.hypot(agent_x - ideal_x, agent_y - ideal_y)
                    max_dist = 100.0
                    defensive_quality = max(0, 1.0 - (dist_from_ideal / max_dist))
                    rewards[i] += w_defense * defensive_quality
                
                # Center coverage
                if w_coverage > 0 and agent_x < half_line * 0.6:  # In defensive third
                    dist_from_goal_center = abs(agent_y - goal_center_y)
                    max_coverage_dist = self.engine.height / 3.0
                    coverage_quality = max(0, 1.0 - (dist_from_goal_center / max_coverage_dist))
                    rewards[i] += w_coverage * coverage_quality
        
        # === 8. PENALTIES ===
        w_collision = reward_weights.get('teammate_collision', 0.0)
        w_action = reward_weights.get('action_penalty', 0.0)
        w_unnecessary = reward_weights.get('unnecessary_movement', 0.0)
        
        for i in range(self.num_team_a):
            agent_x = new_state['agent_x'][i]
            agent_y = new_state['agent_y'][i]
            prev_agent_x = prev_state['agent_x'][i]
            prev_agent_y = prev_state['agent_y'][i]
            
            # Teammate collision
            if w_collision > 0:
                for j in range(i + 1, self.num_team_a):
                    teammate_x = new_state['agent_x'][j]
                    teammate_y = new_state['agent_y'][j]
                    dist_to_teammate = np.hypot(agent_x - teammate_x, agent_y - teammate_y)
                    collision_threshold = paddle_radius * 2 + 10
                    if dist_to_teammate < collision_threshold:
                        collision_penalty = w_collision * (1.0 - dist_to_teammate / collision_threshold)
                        rewards[i] -= collision_penalty
                        rewards[j] -= collision_penalty
            
            # Action penalty
            if w_action > 0:
                action_magnitude = np.linalg.norm(actions[i])
                rewards[i] -= w_action * action_magnitude
            
            # Unnecessary movement
            if w_unnecessary > 0:
                dist_to_puck = np.hypot(agent_x - puck_x, agent_y - puck_y)
                movement = np.hypot(agent_x - prev_agent_x, agent_y - prev_agent_y)
                if dist_to_puck > 100 and not puck_in_defensive_zone:
                    rewards[i] -= w_unnecessary * movement
        
        return rewards
    
    def _check_done(self, state: Dict) -> np.ndarray:
        """Check if episode is done"""
        score_done = (state['team_a_score'] >= self.max_score or 
                     state['team_b_score'] >= self.max_score)
        time_done = self.engine.tick >= self.max_steps
        
        done = score_done or time_done
        
        return np.array([done] * self.num_team_a, dtype=np.float32)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, config: PPOConfig, env: AirHockeyEnv):
        super().__init__()
        self.config = config
        
        obs_dim = env.obs_dim
        hidden = config.hidden_dim
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, env.action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, env.action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def reset_value_head(self):
        """Reset value head (for curriculum phase transitions)"""
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.5)
                nn.init.constant_(layer.bias, 0.0)
        print("✓ Value head reset")
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        features = self.shared_net(obs)
        action_mean = torch.tanh(self.actor_mean(features))
        value = self.critic(features)
        return action_mean, value
    
    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None):
        """Get action, log prob, entropy, and value"""
        action_mean, value = self(obs)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        
        return action, log_prob, entropy, value


class PPOTrainer:
    """PPO training algorithm"""
    
    def __init__(self, config: PPOConfig, scenario_params: Dict):
        self.config = config
        self.device = torch.device(config.device)

        # Store reward weights (from config)
        self.reward_weights = getattr(config, 'reward_weights', None)
        
        # Store opponent type
        self.opponent_type = getattr(config, 'opponent_type', 'static')

        # Create environment
        self.env = AirHockeyEnv(config, scenario_params)
        
        # Update env settings from config
        if hasattr(config, 'max_score'):
            self.env.max_score = config.max_score
        if hasattr(config, 'max_steps'):
            self.env.max_steps = config.max_steps

        # Create actor-critic network
        self.agent = ActorCritic(config, self.env).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.learning_rate)
        
        # Storage for rollouts
        self.reset_rollout_buffer()
        
        # Logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.log_dir}/training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def reset_rollout_buffer(self):
        """Reset storage buffers"""
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.logprob_buffer = []
        self.done_buffer = []
    
    def collect_rollouts(self) -> Dict:
        """Collect rollouts"""
        self.reset_rollout_buffer()
        # DEBUG: Print to verify rewards come from config
        print(f"\n🔍 Phase: {self.current_phase.name}")
        print(f"🔍 Reward weights being used:")
        for key, value in self.current_phase.rewards.items():
            print(f"   {key}: {value}")
        print()
        obs = self.env.reset()
        
        episode_rewards = []
        episode_lengths = []
        episode_goals_for = []
        episode_goals_against = []
        
        current_episode_reward = 0
        current_episode_length = 0
        prev_scores = {'team_a_score': 0, 'team_b_score': 0}
        
        for step in range(self.config.n_steps):
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(obs_tensor)
            
            # Step environment with reward weights
            next_obs, reward, done, info = self.env.step(
                action.cpu().numpy(),
                reward_weights=self.reward_weights  # ← Pass weights from config
            )
            
            # Store transition
            self.obs_buffer.append(obs)
            self.action_buffer.append(action.cpu().numpy())
            self.reward_buffer.append(reward)
            self.value_buffer.append(value.cpu().numpy())
            self.logprob_buffer.append(logprob.cpu().numpy())
            self.done_buffer.append(done)
            
            current_episode_reward += reward.sum()
            current_episode_length += 1
            
            if done.any():
                goals_for = info['scores']['team_a_score'] - prev_scores['team_a_score']
                goals_against = info['scores']['team_b_score'] - prev_scores['team_b_score']
                
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                episode_goals_for.append(goals_for)
                episode_goals_against.append(goals_against)
                
                current_episode_reward = 0
                current_episode_length = 0
                prev_scores = {'team_a_score': 0, 'team_b_score': 0}
                obs = self.env.reset()
            else:
                obs = next_obs
                if info['scores']['team_a_score'] > prev_scores['team_a_score']:
                    prev_scores['team_a_score'] = info['scores']['team_a_score']
                if info['scores']['team_b_score'] > prev_scores['team_b_score']:
                    prev_scores['team_b_score'] = info['scores']['team_b_score']
        
        self._compute_gae()
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_goals_for': episode_goals_for if episode_goals_for else [0],
            'episode_goals_against': episode_goals_against if episode_goals_against else [0],
            'mean_value': np.mean([v.mean() for v in self.value_buffer])
        }
    
    def _compute_gae(self):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(self.reward_buffer)
        last_gae_lam = 0
        
        for t in reversed(range(len(self.reward_buffer))):
            if t == len(self.reward_buffer) - 1:
                next_non_terminal = 1.0 - self.done_buffer[t]
                next_values = self.value_buffer[t]
            else:
                next_non_terminal = 1.0 - self.done_buffer[t]
                next_values = self.value_buffer[t + 1]
            
            current_values = self.value_buffer[t]
            
            delta = (self.reward_buffer[t] + 
                    self.config.gamma * next_values.flatten() * next_non_terminal - 
                    current_values.flatten())
            
            advantages[t] = last_gae_lam = (
                delta + self.config.gamma * self.config.gae_lambda * 
                next_non_terminal * last_gae_lam
            )
        
        self.advantages = advantages
        self.returns = advantages + np.array([v.flatten() for v in self.value_buffer])
    
    def update_policy(self) -> Dict:
        """Update policy using PPO"""
        # Flatten buffers
        obs = np.array(self.obs_buffer).reshape(-1, self.env.obs_dim)
        actions = np.array(self.action_buffer).reshape(-1, self.env.action_dim)
        old_logprobs = np.array(self.logprob_buffer).reshape(-1, 1)
        advantages = self.advantages.reshape(-1, 1)
        returns = self.returns.reshape(-1, 1)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_logprobs_tensor = torch.FloatTensor(old_logprobs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        # PPO epochs
        for epoch in range(self.config.n_epochs):
            indices = np.arange(len(obs))
            np.random.shuffle(indices)
            
            for start in range(0, len(obs), self.config.batch_size):
                end = start + self.config.batch_size
                mb_indices = indices[start:end]
                
                # Get current predictions
                _, new_logprobs, entropy, values = self.agent.get_action_and_value(
                    obs_tensor[mb_indices],
                    actions_tensor[mb_indices]
                )
                
                # Policy loss (PPO clip)
                ratio = torch.exp(new_logprobs - old_logprobs_tensor[mb_indices])
                surr1 = ratio * advantages_tensor[mb_indices]
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 
                                   1 + self.config.clip_range) * advantages_tensor[mb_indices]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.config.clip_range_vf is not None:
                    values_clipped = returns_tensor[mb_indices] + torch.clamp(
                        values - returns_tensor[mb_indices],
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf
                    )
                    value_loss1 = (values - returns_tensor[mb_indices]).pow(2)
                    value_loss2 = (values_clipped - returns_tensor[mb_indices]).pow(2)
                    value_loss = torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = (values - returns_tensor[mb_indices]).pow(2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.vf_coef * value_loss + 
                       self.config.ent_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                n_updates += 1
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }
    
    def save_checkpoint(self, rollout: int):
        """Save checkpoint"""
        checkpoint = {
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'rollout': rollout
        }
        path = f"{self.config.checkpoint_dir}/ppo_checkpoint_{rollout}.pt"
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting PPO training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Total timesteps: {self.config.total_timesteps:,}")
        
        n_rollouts = self.config.total_timesteps // self.config.n_steps
        
        for rollout in range(n_rollouts):
            # Collect experience
            rollout_info = self.collect_rollouts()
            
            # Update policy
            update_info = self.update_policy()
            
            # Logging
            if rollout % self.config.log_interval == 0:
                timesteps = (rollout + 1) * self.config.n_steps
                
                if rollout_info['episode_rewards']:
                    mean_reward = np.mean(rollout_info['episode_rewards'])
                    mean_length = np.mean(rollout_info['episode_lengths'])
                    mean_goals_for = np.mean(rollout_info['episode_goals_for'])
                    mean_goals_against = np.mean(rollout_info['episode_goals_against'])
                else:
                    mean_reward = 0.0
                    mean_length = 0.0
                    mean_goals_for = 0.0
                    mean_goals_against = 0.0
                
                self.logger.info(
                    f"Rollout {rollout}/{n_rollouts} | "
                    f"Steps: {timesteps:,} | "
                    f"Episodes: {rollout_info['num_episodes_completed']} | "
                    f"Reward: {mean_reward:.2f} | "
                    f"Length: {mean_length:.1f} | "
                    f"Goals: {mean_goals_for:.2f}-{mean_goals_against:.2f} | "
                    f"PG Loss: {update_info['policy_loss']:.4f} | "
                    f"V Loss: {update_info['value_loss']:.4f} | "
                    f"Entropy: {update_info['entropy']:.4f}"
                )
            
            # Save checkpoint
            if rollout % self.config.save_interval == 0:
                self.save_checkpoint(rollout)
        
        self.logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/ppo_curriculum.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--mode', type=str, default='standard', 
                       help='Training mode: standard, phase_1, phase_2, etc.')
    args = parser.parse_args()
    
    # Load unified config
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Get shared config (base values)
    shared = full_config.get('shared', {})
    
    # Get mode-specific config
    if args.mode not in full_config:
        print(f"Warning: Mode '{args.mode}' not found in config, using 'standard'")
        args.mode = 'standard'
    
    mode_config = full_config.get(args.mode, {})
    
    # Merge: start with shared, override with mode-specific
    config_dict = {**shared}
    
    # Map mode_config keys to PPOConfig keys
    key_mapping = {
        'timesteps': 'total_timesteps',  # Curriculum phases use 'timesteps'
        'total_timesteps': 'total_timesteps',  # Standard uses 'total_timesteps'
    }
    
    # Update config_dict with mode-specific values
    for mode_key, mode_value in mode_config.items():
        # Map the key if needed
        config_key = key_mapping.get(mode_key, mode_key)
        
        # Only override if it's a PPOConfig parameter
        if config_key in ['total_timesteps', 'learning_rate', 'ent_coef', 
                          'clip_range', 'batch_size', 'n_epochs', 'n_steps',
                          'gamma', 'gae_lambda', 'clip_range_vf', 'vf_coef',
                          'max_grad_norm', 'num_agents_team_a', 'num_agents_team_b',
                          'device', 'hidden_dim', 'log_interval', 'save_interval',
                          'checkpoint_dir', 'log_dir']:
            config_dict[config_key] = mode_value
    
    # Create config object
    config = PPOConfig(**config_dict)
    
    # Store additional training params (not in PPOConfig dataclass)
    config.max_score = mode_config.get('max_score', 3)
    config.max_steps = mode_config.get('max_steps', 1200)
    config.opponent_type = mode_config.get('opponent_type', 'static')
    config.reward_weights = mode_config.get('rewards', {})
    
    # Scenario parameters
    scenario_params = {
        'width': 800,
        'height': 600,
        'goal_gap': 240,
        'goal_offset': 40,
        'unit_speed_px': 4,
        'paddle_radius': 20,
        'puck_radius': 12,
        'puck_max_speed': 6,
    }
    
    # Create trainer
    trainer = PPOTrainer(config, scenario_params)
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=trainer.device, 
                              weights_only=False)
        trainer.agent.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Loaded checkpoint: {args.checkpoint}")
    
    # Print configuration for verification
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Opponent: {config.opponent_type}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Entropy coef: {config.ent_coef}")
    print(f"Clip range: {config.clip_range}")
    print(f"Max score: {config.max_score}")
    print(f"Max steps: {config.max_steps}")
    print(f"{'='*60}\n")
    
    trainer.train()

if __name__ == "__main__":
    main()