"""
Curriculum Learning PPO Trainer - FIXED VERSION

Key improvements:
1. Reward normalization (prevents value explosion)
2. Value function reset at phase transitions
3. KL divergence monitoring (early stopping)
4. Gradient monitoring
5. Enhanced diagnostics

Usage:
    python train_ppo_curriculum.py
"""

import os
import yaml
import torch
import torch.nn as nn
import logging
from pathlib import Path
import sys
import numpy as np
from collections import deque

# Fix import paths
current_file = Path(__file__).resolve()
training_dir = current_file.parent
project_root = training_dir.parent

sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(training_dir))

from train_ppo import PPOTrainer, PPOConfig, AirHockeyEnv


# ============================================
# REWARD NORMALIZER (Critical for stability!)
# ============================================

class RunningMeanStd:
    """
    Tracks running mean and std to normalize returns
    Prevents value function explosion
    """
    def __init__(self, epsilon=1e-8, clip_range=10.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon
        self.clip_range = clip_range
    
    def update(self, x: np.ndarray):
        """Update running statistics"""
        if len(x) == 0:
            return
        
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count if total_count > 0 else 1.0
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize and clip values"""
        if self.count < 2:
            return np.clip(x, -self.clip_range, self.clip_range)
        
        std = np.sqrt(self.var + self.epsilon)
        normalized = (x - self.mean) / std
        return np.clip(normalized, -self.clip_range, self.clip_range)


# ============================================
# CURRICULUM PHASE
# ============================================

class CurriculumPhase:
    """Represents one phase of curriculum learning"""
    
    def __init__(self, name: str, config: dict, base_config: PPOConfig):
        self.name = name
        self.timesteps = config['timesteps']
        self.opponent_type = config.get('opponent_type', 'random')
        self.max_score = config.get('max_score', 3)
        self.max_steps = config.get('max_steps', 1200)
        self.learning_rate = config.get('learning_rate', base_config.learning_rate)
        self.ent_coef = config.get('ent_coef', base_config.ent_coef)
        self.clip_range = config.get('clip_range', base_config.clip_range)
        
        # Read phase-specific team sizes
        self.num_agents_team_b = config.get('num_agents_team_b', base_config.num_agents_team_b)
        
        self.rewards = config.get('rewards', {})


# ============================================
# CURRICULUM TRAINER
# ============================================

class CurriculumTrainer:
    """Manages curriculum learning across multiple phases"""
    
    def __init__(self, curriculum_config_path: str, scenario_params: dict, 
                 load_checkpoint: str = None):
        with open(curriculum_config_path, 'r') as f:
            self.curriculum = yaml.safe_load(f)
        
        self.scenario_params = scenario_params
        
        # Build phases
        shared = self.curriculum.get('shared', {})
        self.base_config = PPOConfig(**shared)
        
        self.phases = []
        phase_keys = [k for k in self.curriculum.keys() if k.startswith('phase_')]
        phase_keys.sort()
        
        for key in phase_keys:
            phase_config = self.curriculum[key]
            phase = CurriculumPhase(
                phase_config['name'],
                phase_config,
                self.base_config
            )
            self.phases.append(phase)
        
        self.load_checkpoint = load_checkpoint
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        os.makedirs(self.base_config.log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.base_config.log_dir}/curriculum_training.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def train(self):
        """Run curriculum learning across all phases"""
        self.logger.info("=" * 60)
        self.logger.info("CURRICULUM LEARNING STARTED")
        self.logger.info("=" * 60)
        
        trainer = None
        total_timesteps_so_far = 0
        
        for phase_idx, phase in enumerate(self.phases):
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info(f"PHASE {phase_idx + 1}/{len(self.phases)}: {phase.name.upper()}")
            self.logger.info(f"Timesteps: {phase.timesteps:,}")
            self.logger.info(f"Opponent: {phase.opponent_type}")
            self.logger.info(f"LR: {phase.learning_rate}, Entropy: {phase.ent_coef}")
            self.logger.info("=" * 60)
            
            # Create or update trainer
            if trainer is None:
                # First phase: create trainer
                trainer = CurriculumPPOTrainer(
                    self.base_config,
                    self.scenario_params,
                    phase
                )
                
                # Load checkpoint if provided
                if self.load_checkpoint:
                    self.logger.info(f"Loading checkpoint: {self.load_checkpoint}")
                    try:
                        checkpoint = torch.load(self.load_checkpoint, 
                                              map_location=trainer.device, 
                                              weights_only=False)
                        
                        if 'model_state_dict' in checkpoint:
                            trainer.agent.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            trainer.agent.load_state_dict(checkpoint)
                        
                        if 'optimizer_state_dict' in checkpoint:
                            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        
                        self.logger.info("✓ Checkpoint loaded successfully!")
                    except Exception as e:
                        self.logger.error(f"Failed to load checkpoint: {e}")
                        self.logger.info("Starting with random initialization")
            else:
                # Update existing trainer for new phase
                trainer.update_for_phase(phase)
            
            # Train this phase
            trainer.train_phase(phase.timesteps)
            
            total_timesteps_so_far += phase.timesteps
            
            # Save checkpoint after phase
            checkpoint_name = f"phase_{phase_idx + 1}_{phase.name}"
            trainer.save_checkpoint_named(checkpoint_name)
            
            self.logger.info(f"Phase {phase_idx + 1} complete. Total timesteps: {total_timesteps_so_far:,}")
        
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("CURRICULUM LEARNING COMPLETED!")
        self.logger.info(f"Total timesteps: {total_timesteps_so_far:,}")
        self.logger.info("=" * 60)


# ============================================
# CURRICULUM PPO TRAINER (with all fixes)
# ============================================

class CurriculumPPOTrainer(PPOTrainer):
    """Extended PPO trainer with curriculum learning and stability fixes"""
    
    def __init__(self, config: PPOConfig, scenario_params: dict, initial_phase: CurriculumPhase):
        super().__init__(config, scenario_params)
        self.current_phase = initial_phase
        self.phase_rollout_count = 0
        self.prev_actions = None
        
        # === CRITICAL: Add reward normalizer ===
        self.reward_normalizer = RunningMeanStd(clip_range=10.0)
        
        # Diagnostics
        self.diagnostics = {
            'puck_contacts': deque(maxlen=100),
            'approach_rewards': deque(maxlen=100),
            'value_estimates': deque(maxlen=100)
        }
        
        self.update_for_phase(initial_phase)
    
    def update_for_phase(self, phase: CurriculumPhase):
        """
        FIXED: Update trainer settings for new curriculum phase
        Includes value function reset and reward normalizer reset
        """
        prev_phase_name = self.current_phase.name if hasattr(self, 'current_phase') else None
        self.current_phase = phase
        self.phase_rollout_count = 0
        self.prev_actions = None
        
        # Update episode termination
        self.env.max_score = phase.max_score
        self.env.max_steps = phase.max_steps
        
        # NEW: Update team sizes if changed
        if hasattr(phase, 'num_agents_team_b'):
            old_team_b = self.config.num_agents_team_b
            new_team_b = phase.num_agents_team_b
            
            if old_team_b != new_team_b:
                self.logger.info(f"  Updating Team B size: {old_team_b} → {new_team_b}")
                self.config.num_agents_team_b = new_team_b
                self.env.num_team_b = new_team_b
                self.env.num_agents = self.env.num_team_a + self.env.num_team_b
                # Reinitialize observation dimension
                self.env.obs_dim = self.env._get_obs_dim()
                
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = phase.learning_rate
        
        # Update entropy and clipping
        self.config.ent_coef = phase.ent_coef
        self.config.clip_range = phase.clip_range
        
        # === CRITICAL FIX 1: Reset value head at phase transition ===
        if prev_phase_name is not None and prev_phase_name != phase.name:
            if hasattr(self.agent, 'reset_value_head'):
                self.agent.reset_value_head()
                self.logger.info("✓ Value head reset for new phase")
            
            # === CRITICAL FIX 2: Reset reward normalizer ===
            self.reward_normalizer = RunningMeanStd(clip_range=10.0)
            self.logger.info("✓ Reward normalizer reset")
        
        self.logger.info(f"Updated for phase: {phase.name}")
        self.logger.info(f"  LR: {phase.learning_rate}, Entropy: {phase.ent_coef}")
        self.logger.info(f"  Max steps: {phase.max_steps}, Max score: {phase.max_score}")
    
    def update_policy(self):
        """
        FIXED: PPO update with numerical stability improvements
        """
        # Flatten buffers
        obs = np.array(self.obs_buffer).reshape(-1, self.env.obs_dim)
        actions = np.array(self.action_buffer).reshape(-1, self.env.action_dim)
        old_logprobs = np.array(self.logprob_buffer).reshape(-1, 1)
        advantages = self.advantages.reshape(-1, 1)
        returns = self.returns.reshape(-1, 1)
        
        # === FIX 3: Normalize advantages ===
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # === FIX 4: Normalize returns (prevents value explosion) ===
        self.reward_normalizer.update(returns.flatten())
        returns_normalized = self.reward_normalizer.normalize(returns)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_logprobs_tensor = torch.FloatTensor(old_logprobs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns_normalized).to(self.device)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clipfrac = 0
        total_kl = 0
        n_updates = 0
        
        # === FIX 5: Early stopping on KL divergence ===
        target_kl = 0.03  # Less aggressive early stopping (increased from 0.02)
        
        for epoch in range(self.config.n_epochs):
            indices = np.arange(len(obs))
            np.random.shuffle(indices)
            
            epoch_kls = []
            
            for start in range(0, len(obs), self.config.batch_size):
                end = start + self.config.batch_size
                mb_indices = indices[start:end]
                
                # Get current predictions
                _, new_logprobs, entropy, values = self.agent.get_action_and_value(
                    obs_tensor[mb_indices],
                    actions_tensor[mb_indices]
                )
                
                # === FIX 6: Monitor KL divergence ===
                logratio = new_logprobs - old_logprobs_tensor[mb_indices]
                ratio = torch.exp(logratio)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    epoch_kls.append(approx_kl.item())
                
                # Policy loss (PPO clip)
                surr1 = ratio * advantages_tensor[mb_indices]
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 
                                   1 + self.config.clip_range) * advantages_tensor[mb_indices]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # === FIX 7: Clipped value loss ===
                value_loss_unclipped = (values - returns_tensor[mb_indices]).pow(2)
                
                if hasattr(self.config, 'clip_range_vf') and self.config.clip_range_vf:
                    values_clipped = returns_tensor[mb_indices] + torch.clamp(
                        values - returns_tensor[mb_indices],
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf
                    )
                    value_loss_clipped = (values_clipped - returns_tensor[mb_indices]).pow(2)
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = value_loss_unclipped.mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.vf_coef * value_loss + 
                       self.config.ent_coef * entropy_loss)
                
                # === FIX 8: Gradient clipping with monitoring ===
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), 
                                                      self.config.max_grad_norm)
                
                # Detect gradient issues
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.logger.warning(f"⚠️  NaN/Inf gradient detected! Skipping update.")
                    continue
                
                if grad_norm > 100.0:
                    self.logger.warning(f"⚠️  Large gradient norm: {grad_norm:.2f}")
                
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                clipfrac = ((ratio - 1.0).abs() > self.config.clip_range).float().mean()
                total_clipfrac += clipfrac.item()
                n_updates += 1
            
            # === FIX 9: Early stopping if KL too high ===
            mean_epoch_kl = np.mean(epoch_kls) if epoch_kls else 0.0
            total_kl += mean_epoch_kl
            
            if mean_epoch_kl > target_kl:
                self.logger.info(f"  Early stopping at epoch {epoch+1}/{self.config.n_epochs} "
                               f"(KL={mean_epoch_kl:.4f} > {target_kl})")
                break
        
        # Store diagnostic info
        self.diagnostics['value_estimates'].append(np.mean([v.mean() for v in self.value_buffer]))
        
        return {
            'policy_loss': total_policy_loss / n_updates if n_updates > 0 else 0,
            'value_loss': total_value_loss / n_updates if n_updates > 0 else 0,
            'entropy': total_entropy / n_updates if n_updates > 0 else 0,
            'clipfrac': total_clipfrac / n_updates if n_updates > 0 else 0,
            'approx_kl': total_kl / (epoch + 1),
        }
    
    def train_phase(self, timesteps: int):
        """Train with enhanced diagnostics"""
        n_rollouts = timesteps // self.config.n_steps
        
        for rollout in range(n_rollouts):
            self.phase_rollout_count += 1
            
            # Collect experience
            rollout_info = self.collect_rollouts_with_opponent(
                self.current_phase.opponent_type
            )
            
            # Update policy
            update_info = self.update_policy()
            
            # Enhanced logging
            if rollout % self.config.log_interval == 0:
                phase_timesteps = (rollout + 1) * self.config.n_steps
                
                if rollout_info['episode_rewards']:
                    mean_reward = np.mean(rollout_info['episode_rewards'])
                    mean_length = np.mean(rollout_info['episode_lengths'])
                    mean_goals_for = np.mean(rollout_info['goals_for'])
                    mean_goals_against = np.mean(rollout_info['goals_against'])
                else:
                    mean_reward = 0
                    mean_length = 0
                    mean_goals_for = 0
                    mean_goals_against = 0
                
                # Basic log
                self.logger.info(
                    f"[{self.current_phase.name}] Rollout {rollout + 1}/{n_rollouts} | "
                    f"Steps: {phase_timesteps}/{timesteps} | "
                    f"Reward: {mean_reward:.2f} | "
                    f"Length: {mean_length:.1f} | "
                    f"Goals: {mean_goals_for:.2f}-{mean_goals_against:.2f} | "
                    f"PG Loss: {update_info['policy_loss']:.4f} | "
                    f"V Loss: {update_info['value_loss']:.4f} | "
                    f"Entropy: {update_info['entropy']:.4f}"
                )
                
                # === NEW: Diagnostic warnings ===
                if rollout > 10:  # After warm-up
                    if mean_reward < -1000:
                        self.logger.warning("⚠️  Very negative rewards! Check reward function.")
                    
                    if update_info['value_loss'] > 1000:
                        self.logger.warning("⚠️  Value loss exploding! Check normalization.")
                    
                    if update_info['entropy'] < 0.1:
                        self.logger.warning("⚠️  Very low entropy - policy collapsed!")
                    
                    # Only warn about no goals if we're past Phase 1
                    if (mean_goals_for == 0 and mean_goals_against == 0 and 
                        rollout > 30 and self.current_phase.name != "learn_hitting"):
                        self.logger.warning("⚠️  No goals - agents may not be hitting puck!")
            
            # Save checkpoint during phase
            if rollout % self.config.save_interval == 0 and rollout > 0:
                ckpt_name = f"{self.current_phase.name}_rollout_{self.phase_rollout_count}"
                self.save_checkpoint_named(ckpt_name)
    
    def collect_rollouts_with_opponent(self, opponent_type: str):
        """Collect rollouts against specified opponent type"""
        self.reset_rollout_buffer()
        obs = self.env.reset()
        episode_rewards = []
        episode_lengths = []
        episode_goals_for = []
        episode_goals_against = []
        current_episode_reward = 0
        current_episode_length = 0
        
        self.prev_actions = np.zeros((self.config.num_agents_team_a, 2), dtype=np.float32)
        
        prev_scores = {'team_a_score': 0, 'team_b_score': 0}
        
        for step in range(self.config.n_steps):
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(obs_tensor)
            
            world_state_before = self.env.engine.get_world_state()
            
            # Generate opponent actions
            opponent_actions = self._generate_opponent_actions(opponent_type, world_state_before)
            
            # Combine actions (handle case with no opponents)
            team_a_actions = action.cpu().numpy()
            if opponent_actions.shape[0] > 0:
                all_actions = np.vstack([team_a_actions, opponent_actions])
            else:
                all_actions = team_a_actions
            
            # Step environment
            next_obs, base_reward, done, info = self.env.step(all_actions)
            
            world_state_after = self.env.engine.get_world_state()
            
            # Apply curriculum-specific reward shaping
            shaped_reward = self._compute_curriculum_reward(
                world_state_before,
                world_state_after,
                team_a_actions,
                self.prev_actions,
                base_reward,
                self.current_phase.rewards
            )
            
            # Store transition
            self.obs_buffer.append(obs)
            self.action_buffer.append(action.cpu().numpy())
            self.reward_buffer.append(shaped_reward)
            self.value_buffer.append(value.cpu().numpy())
            self.logprob_buffer.append(logprob.cpu().numpy())
            self.done_buffer.append(done)
            
            current_episode_reward += shaped_reward.sum()
            current_episode_length += 1
            
            self.prev_actions = team_a_actions.copy()
            
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
                self.prev_actions = np.zeros((self.config.num_agents_team_a, 2), dtype=np.float32)
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
            'goals_for': episode_goals_for if episode_goals_for else [0],
            'goals_against': episode_goals_against if episode_goals_against else [0],
            'mean_value': np.mean([v.mean() for v in self.value_buffer])
        }
    
    def _generate_opponent_actions(self, opponent_type: str, world_state: dict) -> np.ndarray:
        """Generate opponent actions"""
        num_opponents = self.config.num_agents_team_b
        
        # Handle case with no opponents
        if num_opponents == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        if opponent_type == "static":
            return np.zeros((num_opponents, 2), dtype=np.float32)
        elif opponent_type == "random":
            return np.random.uniform(-1, 1, (num_opponents, 2)).astype(np.float32)
        elif opponent_type == "simple":
            # Import simple policy
            from air_hockey_ros.policies.simple_policy import SimpleRegionalAgentPolicy
            actions = []
            for i in range(num_opponents):
                agent_idx = self.config.num_agents_team_a + i
                policy = SimpleRegionalAgentPolicy(agent_idx)
                action = policy.compute_action(world_state)
                actions.append(action)
            return np.array(actions, dtype=np.float32)
        elif opponent_type == "mixed":
            # Mix of different opponents
            if np.random.rand() < 0.5:
                return np.random.uniform(-1, 1, (num_opponents, 2)).astype(np.float32)
            else:
                from air_hockey_ros.policies.simple_policy import SimpleRegionalAgentPolicy
                actions = []
                for i in range(num_opponents):
                    agent_idx = self.config.num_agents_team_a + i
                    policy = SimpleRegionalAgentPolicy(agent_idx)
                    action = policy.compute_action(world_state)
                    actions.append(action)
                return np.array(actions, dtype=np.float32)
        else:
            return np.zeros((num_opponents, 2), dtype=np.float32)
    
    def _compute_curriculum_reward(self, world_state_before: dict, world_state_after: dict,
                                   actions: np.ndarray, prev_actions: np.ndarray,
                                   base_reward: np.ndarray, phase_rewards: dict) -> np.ndarray:
        """Apply curriculum-specific reward shaping"""
        # Start with base reward (includes goal, approach, proximity, contact, shooting)
        rewards = base_reward.copy()
        
        # Phase-specific additional rewards (if any)
        # These are minimal since base rewards are already good
        
        return rewards
    
    def save_checkpoint_named(self, name: str):
        """Save named checkpoint"""
        checkpoint = {
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'phase': self.current_phase.name,
            'name': name,
            'reward_normalizer_mean': self.reward_normalizer.mean,
            'reward_normalizer_var': self.reward_normalizer.var,
            'reward_normalizer_count': self.reward_normalizer.count,
        }
        path = f"{self.config.checkpoint_dir}/curriculum_{name}.pt"
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")


# ============================================
# MAIN
# ============================================

def main():
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
    
    # Create curriculum trainer
    trainer = CurriculumTrainer(
        'config/ppo_curriculum.yaml',
        scenario_params
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()