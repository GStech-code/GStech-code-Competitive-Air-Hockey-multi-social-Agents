# src/environments/base_env.py
"""
Enhanced Base Environment for Hierarchical Air Hockey
Extends the original air hockey environment to support hierarchical agent control.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import math
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum

# Import game core components
from .game_core.game2v2 import Game2v2


class PolicyType(Enum):
    """Available policies for low-level agents"""
    DEFENSIVE = "defensive"
    OFFENSIVE = "offensive" 
    PASSING = "passing"
    NEUTRAL = "neutral"  # Default/balanced behavior


class TeamFormation(Enum):
    """Team formation strategies"""
    BALANCED = "balanced"      # One defensive, one offensive
    AGGRESSIVE = "aggressive"  # Both offensive
    DEFENSIVE = "defensive"    # Both defensive
    COORDINATED = "coordinated" # Dynamic coordination


class EnhancedAirHockey2v2Env(gym.Env):
    """
    Enhanced base environment that supports both hierarchical and flat control modes.
    This serves as the foundation for the hierarchical wrapper.
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 60
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        # Configuration
        self.config = config or {}
        self.render_mode = self.config.get('render_mode', None)
        self.max_episode_steps = self.config.get('max_episode_steps', 3600)
        self.max_score = self.config.get('max_score', 5)
        
        # Initialize game
        self.game = Game2v2(render=(self.render_mode == 'human'))
        self.game.max_score = self.max_score
        
        # Agent identifiers
        self.paddle_agents = ['blue_A', 'blue_B', 'red_A', 'red_B']
        self.team_agents = ['blue_team', 'red_team'] 
        self.blue_paddles = ['blue_A', 'blue_B']
        self.red_paddles = ['red_A', 'red_B']
        
        # Environment state
        self.current_step = 0
        self.episode_start_time = 0
        
        # Policy assignment tracking
        self.current_policies = {
            'blue_A': PolicyType.NEUTRAL,
            'blue_B': PolicyType.NEUTRAL,
            'red_A': PolicyType.NEUTRAL,
            'red_B': PolicyType.NEUTRAL
        }
        
        # Team formations
        self.team_formations = {
            'blue_team': TeamFormation.BALANCED,
            'red_team': TeamFormation.BALANCED
        }
        
        # Communication system
        self.communication_history = {
            'blue_team': [],
            'red_team': []
        }
        self.max_comm_history = self.config.get('max_comm_history', 5)
        
        # Performance tracking
        self.performance_metrics = {
            'goals_scored': {'blue': 0, 'red': 0},
            'shots_taken': {'blue': 0, 'red': 0},
            'passes_completed': {'blue': 0, 'red': 0},
            'defensive_saves': {'blue': 0, 'red': 0},
            'possession_time': {'blue': 0, 'red': 0}
        }
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Reset environment
        self.reset()
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        
        # Low-level paddle action space (movement)
        self.paddle_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # High-level team action space (policy assignments + formation)
        self.team_action_space = spaces.Dict({
            'policy_assignments': spaces.MultiDiscrete([4, 4]),  # 4 policies for each paddle
            'formation_command': spaces.Discrete(4),  # 4 formation types
            'priority_target': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),  # x,y priority
            'communication': spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)  # message vector
        })
        
        # Observation spaces will be defined based on agent type
        self.paddle_obs_dim = 25  # Local tactical information
        self.team_obs_dim = 40    # Strategic game information
        
        self.paddle_observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.paddle_obs_dim,), dtype=np.float32
        )
        
        self.team_observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.team_obs_dim,), dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset game
        self.game.reset_game()
        self.current_step = 0
        self.episode_start_time = pygame.time.get_ticks()
        
        # Reset policies to neutral
        for agent in self.paddle_agents:
            self.current_policies[agent] = PolicyType.NEUTRAL
        
        # Reset formations
        for team in self.team_agents:
            self.team_formations[team] = TeamFormation.BALANCED
        
        # Clear communication history
        self.communication_history = {team: [] for team in self.team_agents}
        
        # Reset performance metrics
        self.performance_metrics = {
            'goals_scored': {'blue': 0, 'red': 0},
            'shots_taken': {'blue': 0, 'red': 0}, 
            'passes_completed': {'blue': 0, 'red': 0},
            'defensive_saves': {'blue': 0, 'red': 0},
            'possession_time': {'blue': 0, 'red': 0}
        }
        
        return self._get_initial_observations()
    
    def step(self, actions: Dict[str, Any]):
        """
        Execute one step. Actions can be from paddle agents, team agents, or both.
        """
        self.current_step += 1
        
        # Process high-level actions (policy assignments, formations)
        if any(agent in actions for agent in self.team_agents):
            self._process_team_actions(actions)
        
        # Process low-level actions (paddle movements)
        paddle_actions = self._extract_paddle_actions(actions)
        
        # Convert to game format and execute
        blue_decisions, red_decisions = self._convert_actions_to_game_format(paddle_actions)
        
        # Update game
        game_state = self.game.update_one_frame(
            blue_decisions, red_decisions, 
            render=(self.render_mode == 'human')
        )
        
        # Update performance metrics
        self._update_performance_metrics(game_state)
        
        # Get observations for all agent types
        observations = self._get_observations()
        
        # Calculate rewards
        rewards = self._calculate_rewards(game_state)
        
        # Check termination
        terminated = (
            self.game.score1 >= self.game.max_score or 
            self.game.score2 >= self.game.max_score
        )
        truncated = self.current_step >= self.max_episode_steps
        
        info = {
            'game_state': game_state,
            'current_policies': self.current_policies.copy(),
            'team_formations': self.team_formations.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'step': self.current_step
        }
        
        return observations, rewards, terminated, truncated, info
    
    def _process_team_actions(self, actions: Dict[str, Any]):
        """Process high-level team management actions"""
        
        for team_agent in self.team_agents:
            if team_agent not in actions:
                continue
                
            team_action = actions[team_agent]
            
            # Update policy assignments
            if 'policy_assignments' in team_action:
                paddle_list = self.blue_paddles if team_agent == 'blue_team' else self.red_paddles
                policy_assignments = team_action['policy_assignments']
                
                for i, paddle_agent in enumerate(paddle_list):
                    policy_idx = policy_assignments[i]
                    self.current_policies[paddle_agent] = list(PolicyType)[policy_idx]
            
            # Update team formation
            if 'formation_command' in team_action:
                formation_idx = team_action['formation_command']
                self.team_formations[team_agent] = list(TeamFormation)[formation_idx]
            
            # Process communication
            if 'communication' in team_action:
                message = team_action['communication']
                self.communication_history[team_agent].append(message)
                if len(self.communication_history[team_agent]) > self.max_comm_history:
                    self.communication_history[team_agent].pop(0)
    
    def _extract_paddle_actions(self, actions: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract paddle movement actions from the action dictionary"""
        paddle_actions = {}
        
        for paddle_agent in self.paddle_agents:
            if paddle_agent in actions:
                paddle_actions[paddle_agent] = actions[paddle_agent]
            else:
                # Default action based on current policy
                paddle_actions[paddle_agent] = self._get_default_action(paddle_agent)
        
        return paddle_actions
    
    def _get_default_action(self, paddle_agent: str) -> np.ndarray:
        """Generate default action based on current policy assignment"""
        policy = self.current_policies[paddle_agent]
        
        # Get paddle and disc positions
        game_state = self.game.get_game_state()
        
        if paddle_agent == 'blue_A':
            paddle_x = game_state['blueA_paddle_x']
            paddle_y = game_state['blueA_paddle_y']
        elif paddle_agent == 'blue_B':
            paddle_x = game_state['blueB_paddle_x'] 
            paddle_y = game_state['blueB_paddle_y']
        elif paddle_agent == 'red_A':
            paddle_x = game_state['redA_paddle_x']
            paddle_y = game_state['redA_paddle_y']
        else:  # red_B
            paddle_x = game_state['redB_paddle_x']
            paddle_y = game_state['redB_paddle_y']
        
        disc_x = game_state['disc_x']
        disc_y = game_state['disc_y']
        
        # Generate policy-based movement
        if policy == PolicyType.OFFENSIVE:
            # Chase the disc aggressively
            dx = disc_x - paddle_x
            dy = disc_y - paddle_y
            return np.array([np.tanh(dx * 3), np.tanh(dy * 3)], dtype=np.float32)
            
        elif policy == PolicyType.DEFENSIVE:
            # Stay closer to own goal
            if paddle_agent.startswith('blue'):
                target_x = 0.25  # Blue goal area
            else:
                target_x = 0.75  # Red goal area
            target_y = 0.5
            
            dx = target_x - paddle_x
            dy = target_y - paddle_y
            return np.array([np.tanh(dx * 2), np.tanh(dy * 2)], dtype=np.float32)
            
        elif policy == PolicyType.PASSING:
            # Position for team coordination
            # This is simplified - real passing would consider teammate position
            dx = disc_x - paddle_x
            dy = disc_y - paddle_y
            return np.array([np.tanh(dx * 1.5), np.tanh(dy * 1.5)], dtype=np.float32)
            
        else:  # NEUTRAL
            # Balanced behavior
            dx = disc_x - paddle_x
            dy = disc_y - paddle_y
            return np.array([np.tanh(dx * 2), np.tanh(dy * 2)], dtype=np.float32)
    
    def _convert_actions_to_game_format(self, paddle_actions: Dict[str, np.ndarray]) -> Tuple[List[Dict], List[Dict]]:
        """Convert paddle actions to game decision format"""
        
        blue_decisions = []
        red_decisions = []
        
        # Process blue team
        for paddle_agent in self.blue_paddles:
            action = paddle_actions.get(paddle_agent, np.array([0.0, 0.0]))
            decision = {
                'move_x': float(np.clip(action[0], -1.0, 1.0)),
                'move_y': float(np.clip(action[1], -1.0, 1.0))
            }
            blue_decisions.append(decision)
        
        # Process red team
        for paddle_agent in self.red_paddles:
            action = paddle_actions.get(paddle_agent, np.array([0.0, 0.0]))
            decision = {
                'move_x': float(np.clip(action[0], -1.0, 1.0)),
                'move_y': float(np.clip(action[1], -1.0, 1.0))
            }
            red_decisions.append(decision)
        
        return blue_decisions, red_decisions
    
    def _update_performance_metrics(self, game_state: Dict[str, Any]):
        """Update performance metrics based on game state"""
        
        # Update goals
        current_blue_score = game_state['score1']
        current_red_score = game_state['score2']
        
        if current_blue_score > self.performance_metrics['goals_scored']['blue']:
            self.performance_metrics['goals_scored']['blue'] = current_blue_score
        
        if current_red_score > self.performance_metrics['goals_scored']['red']:
            self.performance_metrics['goals_scored']['red'] = current_red_score
        
        # Update possession (simplified - based on disc proximity)
        disc_x = game_state['disc_x']
        
        if disc_x < 0.5:
            self.performance_metrics['possession_time']['blue'] += 1
        else:
            self.performance_metrics['possession_time']['red'] += 1
        
        # Additional metrics can be added based on game events
    
    def _get_initial_observations(self) -> Dict[str, np.ndarray]:
        """Get initial observations after reset"""
        game_state = self.game.get_game_state()
        return self._get_observations(game_state)
    
    def _get_observations(self, game_state: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Get observations for all agents"""
        if game_state is None:
            game_state = self.game.get_game_state()
        
        observations = {}
        
        # Generate observations for paddle agents (tactical level)
        for paddle_agent in self.paddle_agents:
            observations[paddle_agent] = self._get_paddle_observation(paddle_agent, game_state)
        
        # Generate observations for team agents (strategic level)
        for team_agent in self.team_agents:
            observations[team_agent] = self._get_team_observation(team_agent, game_state)
        
        return observations
    
    def _get_paddle_observation(self, paddle_agent: str, game_state: Dict) -> np.ndarray:
        """Get tactical observation for a specific paddle"""
        
        # Determine which paddle and team
        if paddle_agent == 'blue_A':
            paddle_x = game_state['blueA_paddle_x']
            paddle_y = game_state['blueA_paddle_y'] 
            paddle_speed = game_state['blueA_paddle_actual_speed']
            is_blue = True
            paddle_idx = 0
        elif paddle_agent == 'blue_B':
            paddle_x = game_state['blueB_paddle_x']
            paddle_y = game_state['blueB_paddle_y']
            paddle_speed = game_state['blueB_paddle_actual_speed']
            is_blue = True
            paddle_idx = 1
        elif paddle_agent == 'red_A':
            paddle_x = game_state['redA_paddle_x']
            paddle_y = game_state['redA_paddle_y']
            paddle_speed = game_state['redA_paddle_actual_speed'] 
            is_blue = False
            paddle_idx = 0
        else:  # red_B
            paddle_x = game_state['redB_paddle_x']
            paddle_y = game_state['redB_paddle_y']
            paddle_speed = game_state['redB_paddle_actual_speed']
            is_blue = False
            paddle_idx = 1
        
        # Basic paddle state
        obs = [
            paddle_x, paddle_y, paddle_speed
        ]
        
        # Disc information
        obs.extend([
            game_state['disc_x'], 
            game_state['disc_y'],
            game_state['disc_velocity_x'],
            game_state['disc_velocity_y']
        ])
        
        # Teammate information
        if is_blue:
            teammate_x = game_state['blueB_paddle_x'] if paddle_idx == 0 else game_state['blueA_paddle_x']
            teammate_y = game_state['blueB_paddle_y'] if paddle_idx == 0 else game_state['blueA_paddle_y']
        else:
            teammate_x = game_state['redB_paddle_x'] if paddle_idx == 0 else game_state['redA_paddle_x']
            teammate_y = game_state['redB_paddle_y'] if paddle_idx == 0 else game_state['redA_paddle_y']
        
        obs.extend([teammate_x, teammate_y])
        
        # Opponent information
        if is_blue:
            obs.extend([
                game_state['redA_paddle_x'], game_state['redA_paddle_y'],
                game_state['redB_paddle_x'], game_state['redB_paddle_y']
            ])
        else:
            obs.extend([
                game_state['blueA_paddle_x'], game_state['blueA_paddle_y'],
                game_state['blueB_paddle_x'], game_state['blueB_paddle_y']
            ])
        
        # Current policy assignment (one-hot encoded)
        current_policy = self.current_policies[paddle_agent]
        policy_encoding = [0.0] * 4
        policy_encoding[list(PolicyType).index(current_policy)] = 1.0
        obs.extend(policy_encoding)
        
        # Game state
        obs.extend([
            game_state['score1'], game_state['score2'],
            game_state['game_time'] / 1000.0  # normalize time
        ])
        
        # Communication from team manager (recent message)
        team_name = 'blue_team' if is_blue else 'red_team'
        if self.communication_history[team_name]:
            latest_message = self.communication_history[team_name][-1]
            obs.extend(latest_message.tolist())
        else:
            obs.extend([0.0] * 8)  # Empty message
        
        return np.array(obs, dtype=np.float32)
    
    def _get_team_observation(self, team_agent: str, game_state: Dict) -> np.ndarray:
        """Get strategic observation for a team manager"""
        
        is_blue = (team_agent == 'blue_team')
        
        # Team paddle positions and states
        if is_blue:
            team_positions = [
                game_state['blueA_paddle_x'], game_state['blueA_paddle_y'],
                game_state['blueB_paddle_x'], game_state['blueB_paddle_y']
            ]
            team_speeds = [
                game_state['blueA_paddle_actual_speed'],
                game_state['blueB_paddle_actual_speed'] 
            ]
            opponent_positions = [
                game_state['redA_paddle_x'], game_state['redA_paddle_y'],
                game_state['redB_paddle_x'], game_state['redB_paddle_y']
            ]
        else:
            team_positions = [
                game_state['redA_paddle_x'], game_state['redA_paddle_y'],
                game_state['redB_paddle_x'], game_state['redB_paddle_y']
            ]
            team_speeds = [
                game_state['redA_paddle_actual_speed'],
                game_state['redB_paddle_actual_speed']
            ]
            opponent_positions = [
                game_state['blueA_paddle_x'], game_state['blueA_paddle_y'],
                game_state['blueB_paddle_x'], game_state['blueB_paddle_y']
            ]
        
        obs = []
        
        # Team formation analysis
        obs.extend(team_positions)
        obs.extend(team_speeds)
        
        # Team spread (how spread out the team is)
        team_spread = math.sqrt((team_positions[0] - team_positions[2])**2 + 
                               (team_positions[1] - team_positions[3])**2)
        obs.append(team_spread)
        
        # Disc information
        obs.extend([
            game_state['disc_x'],
            game_state['disc_y'], 
            game_state['disc_velocity_x'],
            game_state['disc_velocity_y']
        ])
        
        # Opponent analysis
        obs.extend(opponent_positions)
        
        # Opponent formation
        opp_spread = math.sqrt((opponent_positions[0] - opponent_positions[2])**2 +
                              (opponent_positions[1] - opponent_positions[3])**2)
        obs.append(opp_spread)
        
        # Game state
        obs.extend([
            game_state['score1'], game_state['score2'],
            game_state['game_time'] / 1000.0
        ])
        
        # Current policy assignments for own team (encoded)
        paddle_list = self.blue_paddles if is_blue else self.red_paddles
        for paddle in paddle_list:
            policy_encoding = [0.0] * 4
            current_policy = self.current_policies[paddle]
            policy_encoding[list(PolicyType).index(current_policy)] = 1.0
            obs.extend(policy_encoding)
        
        # Performance metrics
        team_color = 'blue' if is_blue else 'red'
        obs.extend([
            self.performance_metrics['possession_time'][team_color] / max(1, self.current_step),
            self.performance_metrics['goals_scored'][team_color],
            self.performance_metrics['shots_taken'][team_color]
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_rewards(self, game_state: Dict) -> Dict[str, float]:
        """Calculate rewards for all agents"""
        rewards = {}
        
        # Base rewards for paddle agents
        for paddle_agent in self.paddle_agents:
            rewards[paddle_agent] = self._calculate_paddle_reward(paddle_agent, game_state)
        
        # Strategic rewards for team agents  
        for team_agent in self.team_agents:
            rewards[team_agent] = self._calculate_team_reward(team_agent, game_state)
        
        return rewards
    
    def _calculate_paddle_reward(self, paddle_agent: str, game_state: Dict) -> float:
        """Calculate reward for individual paddle agent"""
        
        reward = 0.0
        current_policy = self.current_policies[paddle_agent]
        
        # Basic survival reward
        reward += 0.01
        
        # Policy-specific rewards
        if current_policy == PolicyType.DEFENSIVE:
            # Reward defensive positioning and goal protection
            reward += self._calculate_defensive_reward(paddle_agent, game_state)
        elif current_policy == PolicyType.OFFENSIVE:
            # Reward aggressive play and scoring opportunities
            reward += self._calculate_offensive_reward(paddle_agent, game_state)
        elif current_policy == PolicyType.PASSING:
            # Reward team coordination and passing
            reward += self._calculate_passing_reward(paddle_agent, game_state)
        
        # General performance rewards
        reward += self._calculate_general_paddle_reward(paddle_agent, game_state)
        
        return reward
    
    def _calculate_team_reward(self, team_agent: str, game_state: Dict) -> float:
        """Calculate reward for team management agent"""
        
        reward = 0.0
        is_blue = (team_agent == 'blue_team')
        
        # Strategic coordination reward
        reward += self._calculate_coordination_reward(team_agent, game_state)
        
        # Formation quality reward
        reward += self._calculate_formation_reward(team_agent, game_state)
        
        # Goal scoring/conceding
        if is_blue:
            current_score = game_state['score1']
            opponent_score = game_state['score2']
        else:
            current_score = game_state['score2'] 
            opponent_score = game_state['score1']
        
        # Big rewards for team performance
        team_color = 'blue' if is_blue else 'red'
        prev_goals = self.performance_metrics['goals_scored'][team_color]
        
        if current_score > prev_goals:
            reward += 100.0  # Goal scored
        
        return reward
    
    def _calculate_defensive_reward(self, paddle_agent: str, game_state: Dict) -> float:
        """Calculate reward for defensive policy execution"""
        # Simplified defensive reward calculation
        reward = 0.0
        
        # Reward staying near own goal
        is_blue = paddle_agent.startswith('blue')
        if is_blue:
            goal_x = 0.0
        else:
            goal_x = 1.0
            
        goal_y = 0.5
        
        if paddle_agent.endswith('A'):
            if is_blue:
                paddle_x = game_state['blueA_paddle_x']
                paddle_y = game_state['blueA_paddle_y']
            else:
                paddle_x = game_state['redA_paddle_x'] 
                paddle_y = game_state['redA_paddle_y']
        else:
            if is_blue:
                paddle_x = game_state['blueB_paddle_x']
                paddle_y = game_state['blueB_paddle_y']
            else:
                paddle_x = game_state['redB_paddle_x']
                paddle_y = game_state['redB_paddle_y']
        
        # Distance to goal
        goal_distance = math.sqrt((paddle_x - goal_x)**2 + (paddle_y - goal_y)**2)
        reward += max(0, 1.0 - goal_distance) * 2.0
        
        return reward
    
    def _calculate_offensive_reward(self, paddle_agent: str, game_state: Dict) -> float:
        """Calculate reward for offensive policy execution"""
        reward = 0.0
        
        # Reward being close to disc when playing offensively
        if paddle_agent.endswith('A'):
            if paddle_agent.startswith('blue'):
                paddle_x = game_state['blueA_paddle_x']
                paddle_y = game_state['blueA_paddle_y']
                disc_distance = game_state['blueA_paddle_to_disc_distance']
            else:
                paddle_x = game_state['redA_paddle_x']
                paddle_y = game_state['redA_paddle_y'] 
                disc_distance = game_state['redA_paddle_to_disc_distance']
        else:
            if paddle_agent.startswith('blue'):
                paddle_x = game_state['blueB_paddle_x']
                paddle_y = game_state['blueB_paddle_y']
                disc_distance = game_state['blueB_paddle_to_disc_distance']
            else:
                paddle_x = game_state['redB_paddle_x']
                paddle_y = game_state['redB_paddle_y']
                disc_distance = game_state['redB_paddle_to_disc_distance']
        
        # Reward proximity to disc
        max_distance = math.sqrt(self.game.screen_width**2 + self.game.screen_height**2)
        normalized_distance = disc_distance / max_distance
        reward += (1.0 - normalized_distance) * 3.0
        
        return reward
    
    def _calculate_passing_reward(self, paddle_agent: str, game_state: Dict) -> float:
        """Calculate reward for passing/coordination policy"""
        reward = 0.0
        
        # Reward positioning that enables team coordination
        # This is simplified - real passing reward would be more complex
        
        # Reward maintaining good distance from teammate
        is_blue = paddle_agent.startswith('blue')
        paddle_idx = 0 if paddle_agent.endswith('A') else 1
        
        if is_blue:
            if paddle_idx == 0:
                paddle_x = game_state['blueA_paddle_x']
                paddle_y = game_state['blueA_paddle_y']
                teammate_x = game_state['blueB_paddle_x']
                teammate_y = game_state['blueB_paddle_y']
            else:
                paddle_x = game_state['blueB_paddle_x']
                paddle_y = game_state['blueB_paddle_y']
                teammate_x = game_state['blueA_paddle_x'] 
                teammate_y = game_state['blueA_paddle_y']
        else:
            if paddle_idx == 0:
                paddle_x = game_state['redA_paddle_x']
                paddle_y = game_state['redA_paddle_y']
                teammate_x = game_state['redB_paddle_x']
                teammate_y = game_state['redB_paddle_y']
            else:
                paddle_x = game_state['redB_paddle_x']
                paddle_y = game_state['redB_paddle_y']
                teammate_x = game_state['redA_paddle_x']
                teammate_y = game_state['redA_paddle_y']
        
        # Ideal teammate distance
        teammate_distance = math.sqrt((paddle_x - teammate_x)**2 + (paddle_y - teammate_y)**2)
        ideal_distance = 0.3  # Screen units
        distance_quality = 1.0 - abs(teammate_distance - ideal_distance)
        reward += max(0, distance_quality) * 2.0
        
        return reward
    
    def _calculate_general_paddle_reward(self, paddle_agent: str, game_state: Dict) -> float:
        """General rewards that apply regardless of policy"""
        reward = 0.0
        
        # Reward for disc hits
        is_blue = paddle_agent.startswith('blue')
        
        # Small reward for movement (prevent getting stuck)
        if paddle_agent.endswith('A'):
            if is_blue:
                speed = game_state['blueA_paddle_actual_speed']
            else:
                speed = game_state['redA_paddle_actual_speed']
        else:
            if is_blue:
                speed = game_state['blueB_paddle_actual_speed']
            else:
                speed = game_state['redB_paddle_actual_speed']
        
        reward += min(speed * 0.1, 0.5)  # Small movement reward
        
        return reward
    
    def _calculate_coordination_reward(self, team_agent: str, game_state: Dict) -> float:
        """Calculate team coordination reward"""
        reward = 0.0
        
        # Measure how well the team is coordinating
        is_blue = (team_agent == 'blue_team')
        
        if is_blue:
            paddle1_x = game_state['blueA_paddle_x']
            paddle1_y = game_state['blueA_paddle_y']
            paddle2_x = game_state['blueB_paddle_x'] 
            paddle2_y = game_state['blueB_paddle_y']
        else:
            paddle1_x = game_state['redA_paddle_x']
            paddle1_y = game_state['redA_paddle_y']
            paddle2_x = game_state['redB_paddle_x']
            paddle2_y = game_state['redB_paddle_y']
        
        # Formation quality based on current formation command
        formation = self.team_formations[team_agent]
        
        if formation == TeamFormation.BALANCED:
            # One paddle should be more defensive, one more offensive
            center_x = 0.5 if is_blue else 0.5
            defensive_quality = abs(min(paddle1_x, paddle2_x) - (center_x - 0.2))
            offensive_quality = abs(max(paddle1_x, paddle2_x) - (center_x + 0.2))
            reward += max(0, 1.0 - defensive_quality - offensive_quality) * 3.0
            
        elif formation == TeamFormation.AGGRESSIVE:
            # Both paddles should be forward
            forward_x = 0.6 if is_blue else 0.4
            aggression_quality = 2.0 - abs(paddle1_x - forward_x) - abs(paddle2_x - forward_x)
            reward += max(0, aggression_quality) * 2.0
            
        # Add more formation rewards as needed
        
        return reward
    
    def _calculate_formation_reward(self, team_agent: str, game_state: Dict) -> float:
        """Calculate formation maintenance reward"""
        reward = 0.0
        
        # Reward maintaining good team spacing and positioning
        is_blue = (team_agent == 'blue_team')
        
        if is_blue:
            positions = [
                game_state['blueA_paddle_x'], game_state['blueA_paddle_y'],
                game_state['blueB_paddle_x'], game_state['blueB_paddle_y']
            ]
        else:
            positions = [
                game_state['redA_paddle_x'], game_state['redA_paddle_y'],
                game_state['redB_paddle_x'], game_state['redB_paddle_y']
            ]
        
        # Prevent paddles from clustering too much
        distance = math.sqrt((positions[0] - positions[2])**2 + (positions[1] - positions[3])**2)
        
        if distance < 0.1:  # Too close
            reward -= 2.0
        elif distance > 0.15 and distance < 0.4:  # Good spacing
            reward += 1.0
        elif distance > 0.6:  # Too far apart
            reward -= 1.0
        
        return reward
    
    def render(self):
        """Render the environment"""
        if self.render_mode == 'human':
            # The game handles its own rendering
            pygame.display.flip()
            self.game.clock.tick(self.metadata['render_fps'])
        elif self.render_mode == 'rgb_array':
            # Return RGB array of the current state
            if hasattr(self.game, 'screen'):
                rgb_array = pygame.surfarray.array3d(self.game.screen)
                return rgb_array.transpose((1, 0, 2))
        
        return None
    
    def close(self):
        """Close the environment"""
        if hasattr(self.game, 'close'):
            self.game.close()
        pygame.quit()
    
    # Utility methods for external access
    
    def get_current_policies(self) -> Dict[str, PolicyType]:
        """Get current policy assignments"""
        return self.current_policies.copy()
    
    def get_team_formations(self) -> Dict[str, TeamFormation]:
        """Get current team formations"""
        return self.team_formations.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def set_policy(self, paddle_agent: str, policy: PolicyType):
        """Manually set policy for testing purposes"""
        if paddle_agent in self.paddle_agents:
            self.current_policies[paddle_agent] = policy
    
    def set_team_formation(self, team_agent: str, formation: TeamFormation):
        """Manually set team formation for testing purposes"""
        if team_agent in self.team_agents:
            self.team_formations[team_agent] = formation