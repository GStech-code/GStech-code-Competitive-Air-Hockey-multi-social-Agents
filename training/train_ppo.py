"""
PPO Training for Multi-Agent Air Hockey

This script implements Proximal Policy Optimization (PPO) for training
neural network agents in the air hockey environment.

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

# Import from the project
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # Go up one level to project root
from air_hockey_ros.simulations.base import BaseEngine
from air_hockey_ros.policies.neural_network_policy.multiagent_paddle_net import MultiAgentPaddleNet


@dataclass
class PPOConfig:
    """PPO hyperparameters"""
    # Training
    total_timesteps: int = 1_000_000
    n_steps: int = 2048  # steps per rollout
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
    n_envs: int = 4  # parallel environments
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
        self.prev_actions = np.zeros((self.num_team_a, 2), dtype=np.float32)
        self.action_smooth_factor = 0.0  # Blend 30% of previous action
        
        # Observation/action space info
        self.obs_dim = self._get_obs_dim()
        self.action_dim = 2  # (vx, vy)
        
        # Normalization constants
        self.inv_w = 1.0 / float(scenario_params.get('width', 800))
        self.inv_h = 1.0 / float(scenario_params.get('height', 600))
        puck_max = float(scenario_params.get('puck_max_speed', 6.0))
        unit_speed = float(scenario_params.get('unit_speed_px', 4.0))
        self.inv_v = 1.0 / (max(puck_max, unit_speed) * 1.05)
        
    def _get_obs_dim(self) -> int:
        """Calculate observation dimension per agent"""
        # self_xy(2) + puck_xyvy(4) + teammates*5 + opponents*5
        num_teammates = self.num_team_a - 1  # exclude self
        num_opponents = self.num_team_b
        return 2 + 4 + num_teammates * 5 + num_opponents * 5
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observations"""
        self.engine.reset(self.num_team_a, self.num_team_b)
        self.prev_actions = np.zeros((self.num_team_a, 2), dtype=np.float32)  # Reset smoothing
        return self._get_observations()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Step environment with actions
        
        Args:
            actions: (n_agents_total, 2) array of continuous actions in [-1, 1]
                    First num_team_a rows are Team A, rest are Team B
        
        Returns:
            observations, rewards, dones, info
        """
        # Only smooth Team A's actions (the ones being trained)
        team_a_actions = actions[:self.num_team_a]
        smoothed_team_a = (1 - self.action_smooth_factor) * team_a_actions + \
                        self.action_smooth_factor * self.prev_actions
        self.prev_actions = smoothed_team_a.copy()
        
        # Combine smoothed Team A with unsmoothed Team B (opponents)
        if actions.shape[0] > self.num_team_a:
            team_b_actions = actions[self.num_team_a:]
            all_actions = np.vstack([smoothed_team_a, team_b_actions])
        else:
            all_actions = smoothed_team_a
        
        # Convert continuous actions to discrete commands
        commands = []
        for i, action in enumerate(all_actions):
            vx = self._discretize(action[0])
            vy = self._discretize(action[1])
            commands.append((i, vx, vy))
        
        prev_state = self.engine.get_world_state()
        self.engine.apply_commands(commands)
        self.engine.step()
        new_state = self.engine.get_world_state()
        
        obs = self._get_observations()
        rewards = self._compute_rewards(prev_state, new_state, team_a_actions)  # Only Team A actions
        dones = self._check_done(new_state)
        info = {'scores': self.engine.get_scores()}
        
        return obs, rewards, dones, info
    
    def _discretize(self, val: float) -> int:
        """Convert continuous action to discrete {-1, 0, 1}"""
        # Wider deadzone to reduce jitter
        if val > 0.5:  # Changed from 0.33
            return 1
        elif val < -0.5:
            return -1
        return 0
    
    def _get_observations(self) -> np.ndarray:
        """Get observations for all agents (Team A perspective)"""
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
                        ax[j] - ax[i],  # dx_self
                        ay[j] - ay[i],  # dy_self
                        ax[j] - px,     # dx_puck
                        ay[j] - py,     # dy_puck
                        1.0             # is_teammate
                    ]
                    idx += 5
            
            # Opponent features
            for j in range(self.num_team_a, self.num_agents):
                obs[i, idx:idx+5] = [
                    ax[j] - ax[i],  # dx_self
                    ay[j] - ay[i],  # dy_self
                    ax[j] - px,     # dx_puck
                    ay[j] - py,     # dy_puck
                    0.0             # is_opponent
                ]
                idx += 5
        
        return obs
    
    def _compute_rewards(self, prev_state: Dict, new_state: Dict, actions: np.ndarray) -> np.ndarray:
        """
        Team-based reward structure:
        - Goal rewards: SHARED by all teammates (encourages cooperation)
        - Contact rewards: INDIVIDUAL to hitter (prevents free-riding)
        """
        rewards = np.zeros(self.num_team_a, dtype=np.float32)
        
        # === 1. GOAL REWARDS - SHARED BY ENTIRE TEAM ===
        goal_reward = 0.0
        if new_state['team_a_score'] > prev_state['team_a_score']:
            goal_reward = 1000.0  # Team scored!
        if new_state['team_b_score'] > prev_state['team_b_score']:
            goal_reward = -1000.0  # Team was scored on!
        
        # ALL team members get goal reward (encourages cooperation)
        rewards += goal_reward
        
        # === 2. DETECT PUCK VELOCITY CHANGE ===
        puck_x = new_state['puck_x']
        puck_y = new_state['puck_y']
        prev_puck_vx = prev_state.get('puck_vx', 0.0)
        prev_puck_vy = prev_state.get('puck_vy', 0.0)
        new_puck_vx = new_state['puck_vx']
        new_puck_vy = new_state['puck_vy']
        
        delta_puck_vx = new_puck_vx - prev_puck_vx
        delta_puck_vy = new_puck_vy - prev_puck_vy
        puck_vel_change_magnitude = np.hypot(delta_puck_vx, delta_puck_vy)
        
        if puck_vel_change_magnitude > 0.5:  # Significant change
            hit_distance = self.engine.paddle_radius + self.engine.puck_radius + 15
            
            best_agent = None
            best_alignment = -1.0
            
            for i in range(self.num_team_a):
                prev_agent_x = prev_state['agent_x'][i]
                prev_agent_y = prev_state['agent_y'][i]
                agent_x = new_state['agent_x'][i]
                agent_y = new_state['agent_y'][i]
                
                dist_to_puck = np.hypot(agent_x - puck_x, agent_y - puck_y)
                
                if dist_to_puck < hit_distance:
                    # Agent's movement this step
                    agent_dx = agent_x - prev_agent_x
                    agent_dy = agent_y - prev_agent_y
                    agent_movement = np.hypot(agent_dx, agent_dy)
                    
                    if agent_movement > 0.1:  # Agent actually moved
                        # Normalize agent movement direction
                        agent_dx_norm = agent_dx / agent_movement
                        agent_dy_norm = agent_dy / agent_movement
                        
                        # Normalize puck velocity change direction
                        delta_vx_norm = delta_puck_vx / puck_vel_change_magnitude
                        delta_vy_norm = delta_puck_vy / puck_vel_change_magnitude
                        
                        # Check alignment: does puck move in same direction as agent?
                        alignment = agent_dx_norm * delta_vx_norm + agent_dy_norm * delta_vy_norm
                        
                        if alignment > best_alignment:
                            best_alignment = alignment
                            best_agent = i
            
            # === 3. CONTACT REWARD - ONLY TO HITTING AGENT ===
            if best_agent is not None and best_alignment > 0.3:
                rewards[best_agent] += 20.0  # Only hitter gets contact reward
                
                # === 4. SHOOTING REWARD - ONLY TO HITTING AGENT ===
                if new_puck_vx > 0:  # Moving toward opponent goal
                    shot_power = min(new_puck_vx / self.engine.puck_max_speed, 1.0)
                    rewards[best_agent] += 80.0 * shot_power
        
        return rewards
    
    def _check_done(self, state: Dict) -> np.ndarray:
        """Check if episode is done"""
        max_score = getattr(self, 'max_score', 3)  # Use env attribute or default
        max_steps = getattr(self, 'max_steps', 1200)
        
        score_done = (state['team_a_score'] >= max_score or 
                    state['team_b_score'] >= max_score)
        time_done = self.engine.tick >= max_steps
        
        done = score_done or time_done
        
        return np.array([done] * self.num_team_a, dtype=np.float32)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, config: PPOConfig, env: AirHockeyEnv):
        super().__init__()
        self.config = config
        
        # Shared feature extractor (similar to MultiAgentPaddleNet structure)
        obs_dim = env.obs_dim
        hidden = config.hidden_dim
        
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
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution and value"""
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
        
        # Create environment
        self.env = AirHockeyEnv(config, scenario_params)
        
        # Create actor-critic network
        self.agent = ActorCritic(config, self.env).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.learning_rate)
        
        # Storage for rollouts
        self.reset_rollout_buffer()
        
        # Logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging directories and logger"""
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
        """Collect n_steps of experience"""
        self.reset_rollout_buffer()
        obs = self.env.reset()
        
        # Episode tracking
        episode_rewards = []
        episode_lengths = []
        episode_goals_for = []
        episode_goals_against = []
        
        current_episode_reward = 0
        current_episode_length = 0
        current_episode_goals_a = 0
        current_episode_goals_b = 0
        
        # Track steps where done occurred (for debugging)
        num_resets = 0
        
        for step in range(self.config.n_steps):
            # Convert to tensor
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            # Get action
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(obs_tensor)
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action.cpu().numpy())
            
            # Store transition
            self.obs_buffer.append(obs)
            self.action_buffer.append(action.cpu().numpy())
            self.reward_buffer.append(reward)
            self.value_buffer.append(value.cpu().numpy())
            self.logprob_buffer.append(logprob.cpu().numpy())
            self.done_buffer.append(done)
            
            # Update episode metrics
            current_episode_reward += reward.sum()
            current_episode_length += 1
            
            # Track goals
            if 'scores' in info:
                new_score_a = info['scores']['team_a_score']
                new_score_b = info['scores']['team_b_score']
                
                if new_score_a > current_episode_goals_a:
                    current_episode_goals_a = new_score_a
                if new_score_b > current_episode_goals_b:
                    current_episode_goals_b = new_score_b
            
            # Handle episode termination
            # ALL agents must be done (not just any)
            if done.all():
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                episode_goals_for.append(current_episode_goals_a)
                episode_goals_against.append(current_episode_goals_b)
                
                # Reset for next episode
                current_episode_reward = 0
                current_episode_length = 0
                current_episode_goals_a = 0
                current_episode_goals_b = 0
                num_resets += 1
                
                obs = self.env.reset()
            else:
                obs = next_obs
        
        # Handle incomplete episode at end of rollout
        if current_episode_length > 0:
            # Record partial episode for diagnostics
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            episode_goals_for.append(current_episode_goals_a)
            episode_goals_against.append(current_episode_goals_b)
        
        # Compute advantages and returns
        self._compute_gae()
        
        # Detailed diagnostics
        diagnostics = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_goals_for': episode_goals_for,
            'episode_goals_against': episode_goals_against,
            'mean_value': np.mean([v.mean() for v in self.value_buffer]),
            'num_episodes_completed': num_resets,
            'num_steps_collected': self.config.n_steps,
        }
        
        return diagnostics
    
    def _compute_gae(self):
        """Compute Generalized Advantage Estimation"""
        n_steps = len(self.reward_buffer)
        advantages = np.zeros((n_steps, self.config.num_agents_team_a), dtype=np.float32)
        last_gae_lam = 0
        
        # Get next value estimate
        next_obs = self.obs_buffer[-1]
        with torch.no_grad():
            next_obs_tensor = torch.FloatTensor(next_obs).to(self.device)
            _, next_value = self.agent(next_obs_tensor)
            next_value = next_value.cpu().numpy().flatten()
        
        # Compute GAE
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - self.done_buffer[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.done_buffer[t]
                next_values = self.value_buffer[t + 1].flatten()
            
            current_values = self.value_buffer[t].flatten()
            
            delta = (self.reward_buffer[t] + 
                    self.config.gamma * next_values * next_non_terminal - 
                    current_values)
            
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
            # Mini-batch updates
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
                    values_clipped = values + torch.clamp(
                        values - values,
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
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting PPO training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Total timesteps: {self.config.total_timesteps}")
        
        n_rollouts = self.config.total_timesteps // self.config.n_steps
        
        for rollout in range(n_rollouts):
            # Collect experience
            rollout_info = self.collect_rollouts()
            
            # Update policy
            update_info = self.update_policy()
            
            # Logging
            if rollout % self.config.log_interval == 0:
                timesteps = (rollout + 1) * self.config.n_steps
                
                # Extract stats
                num_completed = rollout_info['num_episodes_completed']
                episode_rewards = rollout_info['episode_rewards']
                episode_lengths = rollout_info['episode_lengths']
                goals_for = rollout_info['episode_goals_for']
                goals_against = rollout_info['episode_goals_against']
                
                # Compute meaningful stats
                if episode_rewards:
                    mean_reward = np.mean(episode_rewards)
                    mean_length = np.mean(episode_lengths)
                    mean_goals_a = np.mean(goals_for)
                    mean_goals_b = np.mean(goals_against)
                else:
                    mean_reward = 0.0
                    mean_length = 0.0
                    mean_goals_a = 0.0
                    mean_goals_b = 0.0
                
                # Log with diagnostics
                self.logger.info(
                    f"Rollout {rollout}/{n_rollouts} | "
                    f"Timesteps: {timesteps} | "
                    f"Episodes: {num_completed} | "
                    f"Reward: {mean_reward:.2f} | "
                    f"Length: {mean_length:.1f} | "
                    f"Goals: {mean_goals_a:.2f}-{mean_goals_b:.2f} | "
                    f"Value: {rollout_info['mean_value']:.2f} | "
                    f"PG Loss: {update_info['policy_loss']:.4f} | "
                    f"V Loss: {update_info['value_loss']:.4f} | "
                    f"Entropy: {update_info['entropy']:.4f}"
                )
                
                # Warning if no episodes completing
                if num_completed == 0:
                    self.logger.warning(
                        f"  ⚠️  No episodes completed in this rollout! "
                        f"Collected {len(episode_rewards)} partial episodes. "
                        f"Consider increasing max episode length or checking done conditions."
                    )
            
            # Save checkpoint
            if rollout % self.config.save_interval == 0:
                self.save_checkpoint(rollout)
        
        self.logger.info("Training completed!")

    def save_checkpoint(self, rollout: int):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"ppo_checkpoint_{rollout}.pt"
        )
        torch.save({
            'rollout': rollout,
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")


def main():
    """Main training script"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/ppo_config.yaml')
    parser.add_argument('--checkpoint', default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Filter out unknown parameters for PPOConfig
        valid_fields = {f.name for f in PPOConfig.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        config = PPOConfig(**filtered_config)
    else:
        config = PPOConfig()
    
    # Scenario parameters (from simple_scenario.yaml)
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
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
        trainer.agent.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint loaded successfully!")
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()