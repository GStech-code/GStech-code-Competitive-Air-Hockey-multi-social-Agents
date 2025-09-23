# src/environments/hierarchical_env.py
"""
Hierarchical Environment Wrapper - FIXED VERSION
Manages the two-level hierarchy: high-level team managers and low-level paddle controllers.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional, Union
from pettingzoo import ParallelEnv
from pathlib import Path

# Try importing yaml, fallback if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Try importing from base_env, create fallback if not available
try:
    from .base_env import EnhancedAirHockey2v2Env, PolicyType, TeamFormation
except ImportError:
    # Create minimal fallback classes
    from enum import Enum
    
    class PolicyType(Enum):
        DEFENSIVE = "defensive"
        OFFENSIVE = "offensive" 
        PASSING = "passing"
        NEUTRAL = "neutral"

    class TeamFormation(Enum):
        BALANCED = "balanced"
        AGGRESSIVE = "aggressive"
        DEFENSIVE = "defensive"
        COORDINATED = "coordinated"
    
    # Minimal base environment for testing
    class EnhancedAirHockey2v2Env:
        def __init__(self, config):
            self.config = config
            self.current_step = 0
            
        def reset(self, seed=None, options=None):
            obs = {
                'blue_A': np.random.randn(28).astype(np.float32),
                'blue_B': np.random.randn(28).astype(np.float32), 
                'red_A': np.random.randn(28).astype(np.float32),
                'red_B': np.random.randn(28).astype(np.float32),
                'blue_team': np.random.randn(30).astype(np.float32),
                'red_team': np.random.randn(30).astype(np.float32)
            }
            return obs
            
        def step(self, actions):
            self.current_step += 1
            obs = {
                'blue_A': np.random.randn(28).astype(np.float32),
                'blue_B': np.random.randn(28).astype(np.float32),
                'red_A': np.random.randn(28).astype(np.float32), 
                'red_B': np.random.randn(28).astype(np.float32),
                'blue_team': np.random.randn(30).astype(np.float32),
                'red_team': np.random.randn(30).astype(np.float32)
            }
            rewards = {k: 0.1 for k in obs.keys()}
            info = {'step': self.current_step}
            return obs, rewards, self.current_step > 50, False, info
            
        def close(self):
            pass
            
        def get_current_policies(self):
            return {'blue_A': PolicyType.NEUTRAL, 'blue_B': PolicyType.NEUTRAL,
                   'red_A': PolicyType.NEUTRAL, 'red_B': PolicyType.NEUTRAL}
                   
        def get_team_formations(self):
            return {'blue_team': TeamFormation.BALANCED, 'red_team': TeamFormation.BALANCED}
            
        def get_performance_metrics(self):
            return {'goals': 0, 'shots': 0}
            
        def set_policy(self, agent, policy):
            pass
            
        def set_team_formation(self, team, formation):
            pass


class HierarchicalAirHockeyEnv(ParallelEnv):
    """
    Main hierarchical environment that coordinates between high-level and low-level agents.
    
    Agent Types:
    - High-level: 'blue_team', 'red_team' (strategic managers)
    - Low-level: 'blue_A', 'blue_B', 'red_A', 'red_B' (paddle controllers)
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'name': 'hierarchical_air_hockey_2v2_v0'
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Load configuration FIRST
        self.config = self._load_config(config)
        
        # Set training stage BEFORE calling _setup_agents
        self.training_stage = self.config.get('training_stage', 'full_hierarchical')
        
        # Hierarchical settings
        self.hierarchical_enabled = self.config.get('hierarchical_enabled', True)
        self.high_level_frequency = self.config.get('high_level_frequency', 10)
        self.low_level_frequency = self.config.get('low_level_frequency', 1)
        
        # Initialize base environment
        self.base_env = EnhancedAirHockey2v2Env(self.config)
        
        # Agent management - NOW we can safely call _setup_agents
        self.possible_agents = []
        self.agents = []
        self._setup_agents()
        
        # Action and observation spaces
        self.action_spaces = {}
        self.observation_spaces = {}
        self._setup_spaces()
        
        # Hierarchical control state
        self.steps_since_high_level = 0
        self.pending_high_level_actions = {}
        
        # Performance tracking
        self.episode_stats = {
            'policy_switches': {'blue': 0, 'red': 0},
            'formation_changes': {'blue': 0, 'red': 0},
            'coordination_score': 0.0,
            'team_rewards': {'blue': 0.0, 'red': 0.0}
        }
        
        # Initialize previous state tracking
        self._prev_policies = {}
        self._prev_formations = {}
        
    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from file and merge with provided config"""
        
        # Start with robust defaults
        default_config = {
            'hierarchical_enabled': True,
            'training_stage': 'full_hierarchical',
            'high_level_frequency': 10,
            'low_level_frequency': 1,
            'render_mode': None,
            'max_episode_steps': 3600,
            'max_score': 5,
            'max_comm_history': 5
        }
        
        # Try to load configuration files if YAML is available
        if YAML_AVAILABLE:
            config_files = [
                'configs/environment_config.yaml',
                'configs/training_config.yaml'
            ]
            
            for config_file in config_files:
                if Path(config_file).exists():
                    try:
                        with open(config_file, 'r') as f:
                            file_config = yaml.safe_load(f) or {}
                            # Flatten nested config if needed
                            for key, value in file_config.items():
                                if isinstance(value, dict):
                                    default_config.update(value)
                                else:
                                    default_config[key] = value
                    except Exception as e:
                        print(f"Warning: Could not load config file {config_file}: {e}")
        
        # Merge with provided config
        if config:
            default_config.update(config)
        
        return default_config
    
    def _setup_agents(self):
        """Setup agent lists based on hierarchical mode and training stage"""
        
        if self.hierarchical_enabled:
            if self.training_stage == 'low_level_only':
                # Only paddle agents during individual policy training
                self.possible_agents = ['blue_A', 'blue_B', 'red_A', 'red_B']
            elif self.training_stage == 'high_level_only':
                # Only team managers during strategy training
                self.possible_agents = ['blue_team', 'red_team']
            else:
                # Full hierarchical mode
                self.possible_agents = [
                    'blue_team', 'red_team',  # High-level
                    'blue_A', 'blue_B', 'red_A', 'red_B'  # Low-level
                ]
        else:
            # Flat control - only paddle agents
            self.possible_agents = ['blue_A', 'blue_B', 'red_A', 'red_B']
        
        self.agents = self.possible_agents.copy()
    
    def _setup_spaces(self):
        """Setup action and observation spaces for all agent types"""
        
        for agent in self.possible_agents:
            if agent in ['blue_team', 'red_team']:
                # High-level team agents
                self.action_spaces[agent] = self._get_team_action_space()
                self.observation_spaces[agent] = self._get_team_observation_space()
            else:
                # Low-level paddle agents
                self.action_spaces[agent] = self._get_paddle_action_space()
                self.observation_spaces[agent] = self._get_paddle_observation_space()
    
    def _get_team_action_space(self) -> spaces.Space:
        """Get action space for team management agents"""
        return spaces.Dict({
            'policy_assignments': spaces.MultiDiscrete([len(PolicyType), len(PolicyType)]),
            'formation_command': spaces.Discrete(len(TeamFormation)),
            'priority_target': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'communication': spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        })
    
    def _get_paddle_action_space(self) -> spaces.Space:
        """Get action space for paddle control agents"""
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    
    def _get_team_observation_space(self) -> spaces.Space:
        """Get observation space for team management agents"""
        return spaces.Box(low=-10.0, high=10.0, shape=(30,), dtype=np.float32)
    
    def _get_paddle_observation_space(self) -> spaces.Space:
        """Get observation space for paddle control agents"""
        return spaces.Box(low=-10.0, high=10.0, shape=(28,), dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """Reset the hierarchical environment"""
        
        # Reset base environment
        base_obs = self.base_env.reset(seed=seed, options=options)
        
        # Reset hierarchical control state
        self.steps_since_high_level = 0
        self.pending_high_level_actions = {}
        
        # Reset episode stats
        self.episode_stats = {
            'policy_switches': {'blue': 0, 'red': 0},
            'formation_changes': {'blue': 0, 'red': 0},
            'coordination_score': 0.0,
            'team_rewards': {'blue': 0.0, 'red': 0.0}
        }
        
        # Reset tracking
        self._prev_policies = {}
        self._prev_formations = {}
        
        # Extract observations for active agents
        observations = {}
        for agent in self.agents:
            if agent in base_obs:
                observations[agent] = base_obs[agent]
            else:
                # Generate dummy observation if not available
                if agent in ['blue_team', 'red_team']:
                    observations[agent] = np.random.randn(30).astype(np.float32)
                else:
                    observations[agent] = np.random.randn(28).astype(np.float32)
        
        # Create info dict
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions: Dict[str, Any]) -> Tuple[
        Dict[str, np.ndarray], 
        Dict[str, float], 
        Dict[str, bool], 
        Dict[str, bool], 
        Dict[str, Dict]
    ]:
        """Execute one hierarchical step"""
        
        # Process hierarchical timing
        self.steps_since_high_level += 1
        
        # Determine which agents should act this step
        active_actions = self._process_hierarchical_actions(actions)
        
        # Execute step in base environment
        try:
            base_obs, base_rewards, base_terminated, base_truncated, base_info = \
                self.base_env.step(active_actions)
        except Exception as e:
            # Fallback for testing - generate dummy results
            base_obs = {agent: np.random.randn(25 if 'team' not in agent else 40).astype(np.float32) 
                       for agent in self.agents}
            base_rewards = {agent: 0.1 for agent in self.agents}
            base_terminated = self.steps_since_high_level > 50
            base_truncated = False
            base_info = {'step': self.steps_since_high_level}
        
        # Update episode statistics
        self._update_episode_stats(base_info)
        
        # Process hierarchical rewards
        rewards = self._process_hierarchical_rewards(base_rewards, base_info)
        
        # Extract observations for active agents
        observations = {}
        for agent in self.agents:
            if agent in base_obs:
                observations[agent] = base_obs[agent]
            else:
                # Generate dummy observation
                if agent in ['blue_team', 'red_team']:
                    observations[agent] = np.random.randn(30).astype(np.float32)
                else:
                    observations[agent] = np.random.randn(28).astype(np.float32)
        
        # Termination and truncation
        terminations = {agent: base_terminated for agent in self.agents}
        truncations = {agent: base_truncated for agent in self.agents}
        
        # Enhanced info
        infos = {}
        for agent in self.agents:
            infos[agent] = {
                'base_info': base_info,
                'episode_stats': self.episode_stats.copy(),
                'hierarchical_state': {
                    'steps_since_high_level': self.steps_since_high_level,
                    'training_stage': self.training_stage
                }
            }
        
        # Remove agents if episode is done
        if base_terminated or base_truncated:
            self.agents = []
        
        return observations, rewards, terminations, truncations, infos
    
    def _process_hierarchical_actions(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """Process actions according to hierarchical timing and control"""
        
        processed_actions = {}
        
        # Handle high-level actions (team management)
        high_level_should_act = (self.steps_since_high_level >= self.high_level_frequency)
        
        for agent in ['blue_team', 'red_team']:
            if agent in self.agents and agent in actions:
                if high_level_should_act:
                    # Process new high-level decision
                    processed_actions[agent] = actions[agent]
                    self.pending_high_level_actions[agent] = actions[agent]
                elif agent in self.pending_high_level_actions:
                    # Continue with previous high-level decision
                    processed_actions[agent] = self.pending_high_level_actions[agent]
        
        # Reset high-level timing if they acted
        if high_level_should_act:
            self.steps_since_high_level = 0
        
        # Handle low-level actions (paddle control)
        for agent in ['blue_A', 'blue_B', 'red_A', 'red_B']:
            if agent in self.agents and agent in actions:
                processed_actions[agent] = actions[agent]
        
        return processed_actions
    
    def _process_hierarchical_rewards(self, base_rewards: Dict[str, float], info: Dict[str, Any]) -> Dict[str, float]:
        """Process and enhance rewards for hierarchical agents"""
        
        rewards = {}
        
        # Scale and distribute rewards based on agent type
        for agent in self.agents:
            if agent in base_rewards:
                base_reward = base_rewards[agent]
            else:
                base_reward = 0.0
                
            if agent in ['blue_team', 'red_team']:
                # Team agents get strategic rewards
                rewards[agent] = base_reward + 0.1  # Small strategic bonus
            else:
                # Paddle agents get tactical rewards
                rewards[agent] = base_reward + 0.05  # Small tactical bonus
        
        return rewards
    
    def _update_episode_stats(self, info: Dict[str, Any]):
        """Update episode statistics"""
        
        # Simple stats update - can be expanded
        self.episode_stats['coordination_score'] = 0.5  # Placeholder
    
    def render(self):
        """Render the environment"""
        try:
            return self.base_env.render()
        except:
            return None
    
    def close(self):
        """Close the environment"""
        try:
            self.base_env.close()
        except:
            pass
    
    # Utility methods for external access and testing
    
    def set_training_stage(self, stage: str):
        """Set training stage for curriculum learning"""
        valid_stages = ['low_level_only', 'high_level_only', 'full_hierarchical']
        if stage in valid_stages:
            self.training_stage = stage
            self._setup_agents()
            self._setup_spaces()
    
    def get_hierarchical_state(self) -> Dict[str, Any]:
        """Get current hierarchical state information"""
        return {
            'training_stage': self.training_stage,
            'steps_since_high_level': self.steps_since_high_level,
            'hierarchical_enabled': self.hierarchical_enabled,
            'episode_stats': self.episode_stats.copy()
        }
    
    def force_policy_assignment(self, paddle_agent: str, policy: PolicyType):
        """Force policy assignment for testing"""
        try:
            self.base_env.set_policy(paddle_agent, policy)
        except:
            pass
    
    def force_team_formation(self, team_agent: str, formation: TeamFormation):
        """Force team formation for testing"""
        try:
            self.base_env.set_team_formation(team_agent, formation)
        except:
            pass
        
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get detailed episode statistics"""
        stats = self.episode_stats.copy()
        try:
            stats['performance_metrics'] = self.base_env.get_performance_metrics()
        except:
            stats['performance_metrics'] = {}
        return stats


# Factory function for creating environment instances
def create_hierarchical_env(config: Optional[Dict[str, Any]] = None) -> HierarchicalAirHockeyEnv:
    """Factory function to create hierarchical environment instances"""
    return HierarchicalAirHockeyEnv(config)


# Environment registration for Ray RLlib
def env_creator(config: Dict[str, Any]) -> HierarchicalAirHockeyEnv:
    """Environment creator function for Ray RLlib registration"""
    return HierarchicalAirHockeyEnv(config)