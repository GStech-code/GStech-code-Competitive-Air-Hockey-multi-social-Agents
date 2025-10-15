"""
Complete Curriculum Learning PPO Trainer
Gradually increases task difficulty over training phases
"""

import os
import yaml
import torch
import logging
from pathlib import Path
import sys
import numpy as np

# Fix import paths
current_file = Path(__file__).resolve()
training_dir = current_file.parent
project_root = training_dir.parent

# Add to path
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(training_dir))

# Import from train_ppo (same directory)
from train_ppo import PPOTrainer, PPOConfig, AirHockeyEnv


class CurriculumPhase:
    """Represents one phase of curriculum learning"""
    
    def __init__(self, name: str, config: dict, base_config: PPOConfig):
        self.name = name
        self.timesteps = config['timesteps']
        self.opponent_type = config.get('opponent_type', 'random')
        self.max_score = config.get('max_score', 3)
        self.max_steps = config.get('max_steps', 1200)
        # Update hyperparameters for this phase
        self.learning_rate = config.get('learning_rate', base_config.learning_rate)
        self.ent_coef = config.get('ent_coef', base_config.ent_coef)
        self.clip_range = config.get('clip_range', base_config.clip_range)
        
        # Reward weights for this phase
        self.rewards = config.get('rewards', {})


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
            
            # Create or update trainer for this phase
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
                        
                        # Load model state
                        if 'model_state_dict' in checkpoint:
                            trainer.agent.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            trainer.agent.load_state_dict(checkpoint)
                        
                        # Load optimizer state if available
                        if 'optimizer_state_dict' in checkpoint:
                            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        
                        self.logger.info("✓ Checkpoint loaded successfully!")
                        
                        # Log checkpoint info if available
                        if 'phase' in checkpoint:
                            self.logger.info(f"  Previous phase: {checkpoint['phase']}")
                        if 'name' in checkpoint:
                            self.logger.info(f"  Checkpoint name: {checkpoint['name']}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to load checkpoint: {e}")
                        self.logger.info("Starting with random initialization instead")
            else:
                # Update existing trainer for new phase
                trainer.update_for_phase(phase)
            
            # Train this phase
            trainer.train_phase(phase.timesteps)
            
            total_timesteps_so_far += phase.timesteps
            
            # Save checkpoint after each phase
            checkpoint_name = f"phase_{phase_idx + 1}_{phase.name}"
            trainer.save_checkpoint_named(checkpoint_name)
            
            self.logger.info(f"Phase {phase_idx + 1} complete. Total timesteps: {total_timesteps_so_far:,}")
        
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("CURRICULUM LEARNING COMPLETED!")
        self.logger.info(f"Total timesteps: {total_timesteps_so_far:,}")
        self.logger.info("=" * 60)


class CurriculumPPOTrainer(PPOTrainer):
    """Extended PPO trainer with curriculum learning support"""
    
    def __init__(self, config: PPOConfig, scenario_params: dict, initial_phase: CurriculumPhase):
        super().__init__(config, scenario_params)
        self.current_phase = initial_phase
        self.phase_rollout_count = 0
        
        # Track previous actions for smoothness rewards
        self.prev_actions = None
        
        self.update_for_phase(initial_phase)
    
    def update_for_phase(self, phase: CurriculumPhase):
        """Update trainer settings for new curriculum phase"""
        self.current_phase = phase
        self.phase_rollout_count = 0
        self.prev_actions = None
        
        # Update episode termination for this phase
        self.env.max_score = phase.max_score
        self.env.max_steps = phase.max_steps

        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = phase.learning_rate
        
        # Update entropy coefficient
        self.config.ent_coef = phase.ent_coef
        self.config.clip_range = phase.clip_range
        
        self.logger.info(f"Updated for phase: {phase.name}")
    
    def train_phase(self, timesteps: int):
        """Train for specified timesteps in current phase"""
        n_rollouts = timesteps // self.config.n_steps
        
        for rollout in range(n_rollouts):
            self.phase_rollout_count += 1
            
            # Collect experience
            rollout_info = self.collect_rollouts_with_opponent(
                self.current_phase.opponent_type
            )
            
            # Update policy
            update_info = self.update_policy()
            
            # Logging
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
        
        # Reset previous actions
        self.prev_actions = np.zeros((self.config.num_agents_team_a, 2), dtype=np.float32)
        
        # Track scores
        prev_scores = {'team_a_score': 0, 'team_b_score': 0}
        
        for step in range(self.config.n_steps):
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(obs_tensor)
            
            # Get current world state
            world_state_before = self.env.engine.get_world_state()
            
            # Get opponent actions based on type
            opponent_actions = self._generate_opponent_actions(opponent_type, world_state_before)
            
            # Combine actions
            team_a_actions = action.cpu().numpy()
            all_actions = np.vstack([team_a_actions, opponent_actions])
            
            # Step environment (uses your improved _compute_rewards)
            next_obs, base_reward, done, info = self.env.step(all_actions)
            
            # Get world state after step
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
            
            # Update previous actions for next step
            self.prev_actions = team_a_actions.copy()
            
            # Check for episode termination
            if done.any():
                # Track goals
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
                # Update score tracking
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
    
    def _compute_curriculum_reward(self, world_state_before: dict, world_state_after: dict,
                                   actions: np.ndarray, prev_actions: np.ndarray,
                                   base_reward: np.ndarray, phase_rewards: dict) -> np.ndarray:
        """
        Complete curriculum reward computation with all shaping terms.
        
        Args:
            world_state_before: State before action
            world_state_after: State after action
            actions: Current actions taken [num_agents, 2]
            prev_actions: Previous actions [num_agents, 2]
            base_reward: Base reward from environment (includes goals)
            phase_rewards: Curriculum phase reward weights
        
        Returns:
            Shaped rewards for each agent
        """
        # Start with base reward (already includes goal rewards from env)
        rewards = base_reward.copy()
        
        # Get phase-specific weights
        w_approach = phase_rewards.get('approach_puck_in_half', 0.0)
        w_close = phase_rewards.get('close_to_puck', 0.0)
        w_puck_vel = phase_rewards.get('puck_velocity_toward_goal', 0.0)
        w_defense = phase_rewards.get('defensive_position', 0.0)
        w_coverage = phase_rewards.get('center_coverage', 0.0)
        w_action = phase_rewards.get('action_penalty', 0.0)
        w_unnecessary = phase_rewards.get('unnecessary_movement', 0.0)
        w_teammate_collision = phase_rewards.get('teammate_collision', 1.0)
        
        # Get positions from after state
        puck_x, puck_y = world_state_after['puck_x'], world_state_after['puck_y']
        prev_puck_x, prev_puck_y = world_state_before['puck_x'], world_state_before['puck_y']
        puck_vx = world_state_after['puck_vx']
        half_line = self.env.engine.width / 2.0
        goal_center_y = self.env.engine.height / 2.0
        paddle_radius = self.env.engine.paddle_radius
        puck_radius = self.env.engine.puck_radius
        hit_distance = paddle_radius + puck_radius + 5
        
        for i in range(self.config.num_agents_team_a):
            agent_x = world_state_after['agent_x'][i]
            agent_y = world_state_after['agent_y'][i]
            prev_agent_x = world_state_before['agent_x'][i]
            prev_agent_y = world_state_before['agent_y'][i]
            
            # Distance to puck
            dist_to_puck = np.hypot(agent_x - puck_x, agent_y - puck_y)
            prev_dist_to_puck = np.hypot(prev_agent_x - prev_puck_x, prev_agent_y - prev_puck_y)
            
            # === 1. APPROACH PUCK (when in our half) ===
            if w_approach > 0 and puck_x < half_line:
                # Reward for moving closer to puck
                if dist_to_puck < prev_dist_to_puck:
                    rewards[i] += w_approach
                elif dist_to_puck > prev_dist_to_puck:
                    rewards[i] -= w_approach * 0.5
            
            # === 2. CLOSE TO PUCK ===
            if w_close > 0 and dist_to_puck < hit_distance:
                rewards[i] += w_close
            
            # === 3. PUCK VELOCITY TOWARD GOAL ===
            if w_puck_vel > 0 and puck_vx > 0 and dist_to_puck < hit_distance * 2:
                # Puck moving right (toward opponent) and we're nearby (we likely hit it)
                rewards[i] += w_puck_vel * abs(puck_vx)
            
            # === 4. DEFENSIVE POSITION ===
            if w_defense > 0:
                # Reward being between puck and our goal
                if agent_x < puck_x:
                    rewards[i] += w_defense
                else:
                    rewards[i] -= w_defense * 0.5
            
            # === 5. CENTER COVERAGE ===
            if w_coverage > 0:
                y_coverage = abs(agent_y - goal_center_y)
                max_y_dist = self.env.engine.height / 4.0
                if y_coverage < max_y_dist:
                    coverage_reward = w_coverage * (1.0 - y_coverage / max_y_dist)
                    rewards[i] += coverage_reward
            
            # === 6. ACTION PENALTY (energy efficiency) ===
            if w_action > 0:
                # Current action magnitude
                action_magnitude = np.abs(actions[i]).sum()
                rewards[i] -= w_action * action_magnitude
                
                # Additional penalty for action changes (jitter reduction)
                if prev_actions is not None:
                    action_change = np.abs(actions[i] - prev_actions[i]).sum()
                    rewards[i] -= w_action * 0.5 * action_change
            
            # === 7. UNNECESSARY MOVEMENT ===
            if w_unnecessary > 0 and dist_to_puck > self.env.engine.width / 3:
                # Agent is far from action but still moving
                action_magnitude = np.abs(actions[i]).sum()
                if action_magnitude > 0.5:  # Threshold for "moving"
                    rewards[i] -= w_unnecessary * action_magnitude
        
        # === 8. TEAMMATE COLLISION PENALTY ===
        if w_teammate_collision > 0:
            for i in range(self.config.num_agents_team_a):
                for j in range(i + 1, self.config.num_agents_team_a):
                    dist_to_teammate = np.hypot(
                        world_state_after['agent_x'][i] - world_state_after['agent_x'][j],
                        world_state_after['agent_y'][i] - world_state_after['agent_y'][j]
                    )
                    collision_threshold = 2 * paddle_radius + 15
                    if dist_to_teammate < collision_threshold:
                        penalty_scale = (1.0 - dist_to_teammate / collision_threshold)
                        collision_penalty = w_teammate_collision * penalty_scale
                        if dist_to_teammate < 2 * paddle_radius:
                            collision_penalty += w_teammate_collision * 2.0
                        rewards[i] -= collision_penalty
                        rewards[j] -= collision_penalty
        
        return rewards
    
    def _generate_opponent_actions(self, opponent_type: str, world_state: dict):
        """Generate opponent actions based on type"""
        num_opponents = self.config.num_agents_team_b
        actions = np.zeros((num_opponents, 2), dtype=np.float32)
        
        if opponent_type == 'static':
            pass  # Already zeros
        
        elif opponent_type == 'random':
            for i in range(num_opponents):
                actions[i, 0] = np.random.uniform(-1, 1)
                actions[i, 1] = np.random.uniform(-1, 1)
        
        elif opponent_type == 'simple':
            # Use simplified defensive logic
            puck_x = world_state['puck_x']
            puck_y = world_state['puck_y']
            width = self.env.engine.width
            height = self.env.engine.height
            
            for i in range(num_opponents):
                agent_idx = self.config.num_agents_team_a + i
                agent_x = world_state['agent_x'][agent_idx]
                agent_y = world_state['agent_y'][agent_idx]
                
                # Flip for Team B perspective
                puck_x_flipped = width - puck_x
                agent_x_flipped = width - agent_x
                
                dx = puck_x_flipped - agent_x_flipped
                dy = puck_y - agent_y
                
                # Simple chase logic
                actions[i, 0] = -np.sign(dx) if abs(dx) > 20 else 0  # Flip back
                actions[i, 1] = np.sign(dy) if abs(dy) > 20 else 0
        
        elif opponent_type == 'mixed':
            # Randomly choose opponent type for variety
            choice = np.random.choice(['static', 'random', 'simple'])
            return self._generate_opponent_actions(choice, world_state)
        
        return actions
    
    def save_checkpoint_named(self, name: str):
        """Save checkpoint with custom name"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"curriculum_{name}.pt"
        )
        torch.save({
            'name': name,
            'phase': self.current_phase.name if self.current_phase else 'unknown',
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")


def main():
    """Main curriculum training script"""
    curriculum_config = "config/ppo_curriculum.yaml"
    
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
    
    trainer = CurriculumTrainer(curriculum_config, scenario_params)
    trainer.train()


if __name__ == "__main__":
    main()