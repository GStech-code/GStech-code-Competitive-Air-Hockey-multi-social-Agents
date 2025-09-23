import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Dict, List, Tuple, Any, Optional
import math

# Import your existing game class
from .game2v2 import Game2v2

class AirHockey2v2Env(gym.Env):
    """
    Gymnasium environment for 2v2 Air Hockey game.
    
    This environment supports 4 agents:
    - blue_A: First blue team paddle
    - blue_B: Second blue team paddle  
    - red_A: First red team paddle
    - red_B: Second red team paddle
    
    The environment can be used for multi-agent reinforcement learning.
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 60
    }
    
    def __init__(self, render_mode: Optional[str] = None, max_episode_steps: int = 1000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Initialize the game
        self.game = Game2v2(render=(render_mode == 'human'))
        
        # Define action space for each paddle
        # Actions: [move_x, move_y] where each is in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # Define observation space
        # Based on get_game_state() method from your Game2v2 class
        obs_low = np.array([
            -1.0,  # paddle_actual_speed (normalized)
            -1.0,  # paddle_x (normalized)
            0.0,   # paddle_y (normalized)  
            0.0,   # paddle_ratio_times_not_moving
            0.0,   # disc_x (normalized)
            0.0,   # disc_y (normalized)
            -1.0,  # disc_velocity_x (normalized)
            -1.0,  # disc_velocity_y (normalized)
            0.0,   # paddle_in_own_goal (boolean as float)
            0.0,   # paddle_num_in_goal
            0.0,   # score1
            0.0,   # score2
            0.0,   # paddle_to_disc_distance
            0.0,   # game_time
            0.0,   # num_blue_hits
            0.0,   # num_red_hits
            # Add other paddles' positions for awareness
            -1.0,  # teammate_x
            0.0,   # teammate_y
            -1.0,  # opponent1_x  
            0.0,   # opponent1_y
            -1.0,  # opponent2_x
            0.0,   # opponent2_y
        ], dtype=np.float32)
        
        obs_high = np.array([
            1.0,   # paddle_actual_speed (normalized)
            1.0,   # paddle_x (normalized)
            1.0,   # paddle_y (normalized)
            1.0,   # paddle_ratio_times_not_moving
            1.0,   # disc_x (normalized)
            1.0,   # disc_y (normalized)
            1.0,   # disc_velocity_x (normalized)
            1.0,   # disc_velocity_y (normalized)
            1.0,   # paddle_in_own_goal (boolean as float)
            100.0, # paddle_num_in_goal (arbitrary high number)
            10.0,  # score1 (max score is typically 5-10)
            10.0,  # score2
            1000.0, # paddle_to_disc_distance (screen diagonal)
            1000.0, # game_time (arbitrary high number)
            1000.0, # num_blue_hits
            1000.0, # num_red_hits
            # Other paddles
            1.0,   # teammate_x
            1.0,   # teammate_y
            1.0,   # opponent1_x
            1.0,   # opponent1_y
            1.0,   # opponent2_x
            1.0,   # opponent2_y
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
        
        # Agent identifiers
        self.agents = ['blue_A', 'blue_B', 'red_A', 'red_B']
        self.blue_agents = ['blue_A', 'blue_B']
        self.red_agents = ['red_A', 'red_B']
        
        # Episode tracking
        self.previous_scores = [0, 0]  # [blue_score, red_score]
        
    def _convert_action(self, action: np.ndarray) -> Dict[str, float]:
        """Convert gymnasium action to game decision format"""
        # Clamp actions to valid range
        action = np.clip(action, -1.0, 1.0)
        
        return {
            'move_x': float(action[0]),
            'move_y': float(action[1])
        }
    
    def _get_observation_for_agent(self, agent_id: str, game_state: Dict) -> np.ndarray:
        """Get observation for a specific agent"""
        
        # Map agent to paddle data
        if agent_id == 'blue_A':
            paddle_speed = game_state['blueA_paddle_actual_speed']
            paddle_x = game_state['blueA_paddle_x'] 
            paddle_y = game_state['blueA_paddle_y']
            paddle_not_moving = game_state['blueA_paddle_ratio_times_not_moving']
            paddle_in_goal = float(game_state['blueA_paddle_in_own_goal'])
            paddle_num_in_goal = game_state['blueA_paddle_num_in_goal']
            paddle_to_disc_dist = game_state['blueA_paddle_to_disc_distance']
            teammate_x = game_state['blueB_paddle_x']
            teammate_y = game_state['blueB_paddle_y'] 
            opponent1_x = game_state['redA_paddle_x']
            opponent1_y = game_state['redA_paddle_y']
            opponent2_x = game_state['redB_paddle_x'] 
            opponent2_y = game_state['redB_paddle_y']
        elif agent_id == 'blue_B':
            paddle_speed = game_state['blueB_paddle_actual_speed']
            paddle_x = game_state['blueB_paddle_x']
            paddle_y = game_state['blueB_paddle_y']
            paddle_not_moving = game_state['blueB_paddle_ratio_times_not_moving']
            paddle_in_goal = float(game_state['blueB_paddle_in_own_goal'])
            paddle_num_in_goal = game_state['blueB_paddle_num_in_goal']
            paddle_to_disc_dist = game_state['blueB_paddle_to_disc_distance']
            teammate_x = game_state['blueA_paddle_x']
            teammate_y = game_state['blueA_paddle_y']
            opponent1_x = game_state['redA_paddle_x']
            opponent1_y = game_state['redA_paddle_y'] 
            opponent2_x = game_state['redB_paddle_x']
            opponent2_y = game_state['redB_paddle_y']
        elif agent_id == 'red_A':
            paddle_speed = game_state['redA_paddle_actual_speed'] 
            paddle_x = game_state['redA_paddle_x']
            paddle_y = game_state['redA_paddle_y']
            paddle_not_moving = game_state['redA_paddle_ratio_times_not_moving']
            paddle_in_goal = float(game_state['redA_paddle_in_own_goal'])
            paddle_num_in_goal = game_state['redA_paddle_num_in_goal']
            paddle_to_disc_dist = game_state['redA_paddle_to_disc_distance']
            teammate_x = game_state['redB_paddle_x']
            teammate_y = game_state['redB_paddle_y']
            opponent1_x = game_state['blueA_paddle_x']
            opponent1_y = game_state['blueA_paddle_y']
            opponent2_x = game_state['blueB_paddle_x']
            opponent2_y = game_state['blueB_paddle_y']
        else:  # red_B
            paddle_speed = game_state['redB_paddle_actual_speed']
            paddle_x = game_state['redB_paddle_x']
            paddle_y = game_state['redB_paddle_y'] 
            paddle_not_moving = game_state['redB_paddle_ratio_times_not_moving']
            paddle_in_goal = float(game_state['redB_paddle_in_own_goal'])
            paddle_num_in_goal = game_state['redB_paddle_num_in_goal']
            paddle_to_disc_dist = game_state['redB_paddle_to_disc_distance']
            teammate_x = game_state['redA_paddle_x']
            teammate_y = game_state['redA_paddle_y']
            opponent1_x = game_state['blueA_paddle_x']
            opponent1_y = game_state['blueA_paddle_y']
            opponent2_x = game_state['blueB_paddle_x']
            opponent2_y = game_state['blueB_paddle_y']
        
        # Create observation vector
        obs = np.array([
            paddle_speed,
            paddle_x,
            paddle_y,
            paddle_not_moving,
            game_state['disc_x'],
            game_state['disc_y'],
            game_state['disc_velocity_x'],
            game_state['disc_velocity_y'],
            paddle_in_goal,
            paddle_num_in_goal,
            game_state['score1'],
            game_state['score2'], 
            paddle_to_disc_dist,
            game_state['game_time'],
            game_state['num_blue_hits'],
            game_state['num_red_hits'],
            teammate_x,
            teammate_y,
            opponent1_x,
            opponent1_y,
            opponent2_x,
            opponent2_y,
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_rewards(self, game_state: Dict) -> Dict[str, float]:
        """Calculate rewards for all agents"""
        rewards = {agent: 0.0 for agent in self.agents}
        
        current_scores = [game_state['score1'], game_state['score2']]
        
        # Scoring rewards
        blue_scored = current_scores[0] > self.previous_scores[0]
        red_scored = current_scores[1] > self.previous_scores[1]
        
        if blue_scored:
            # Blue team gets positive reward for scoring
            for agent in self.blue_agents:
                rewards[agent] += 100.0
            # Red team gets negative reward for being scored on  
            for agent in self.red_agents:
                rewards[agent] -= 100.0
                
        if red_scored:
            # Red team gets positive reward for scoring
            for agent in self.red_agents:
                rewards[agent] += 100.0
            # Blue team gets negative reward for being scored on
            for agent in self.blue_agents:  
                rewards[agent] -= 100.0
        
        # Distance-based reward (encourage moving towards disc)
        disc_x = game_state['disc_x'] * self.game.screen_width
        disc_y = game_state['disc_y'] * self.game.screen_height
        
        # Small reward for being close to disc
        max_distance = math.sqrt(self.game.screen_width**2 + self.game.screen_height**2)
        
        distances = {
            'blue_A': game_state['blueA_paddle_to_disc_distance'],
            'blue_B': game_state['blueB_paddle_to_disc_distance'], 
            'red_A': game_state['redA_paddle_to_disc_distance'],
            'red_B': game_state['redB_paddle_to_disc_distance']
        }
        
        for agent in self.agents:
            # Normalize distance and give small reward for being close
            normalized_dist = distances[agent] / max_distance
            rewards[agent] += (1.0 - normalized_dist) * 0.1
        
        # Penalty for staying in goal too long
        goal_penalties = {
            'blue_A': game_state['blueA_paddle_num_in_goal'],
            'blue_B': game_state['blueB_paddle_num_in_goal'],
            'red_A': game_state['redA_paddle_num_in_goal'], 
            'red_B': game_state['redB_paddle_num_in_goal']
        }
        
        for agent in self.agents:
            if goal_penalties[agent] > 10:  # Arbitrary threshold
                rewards[agent] -= 0.1
        
        # Small reward for hitting the disc (encourage active play)
        if game_state['num_blue_hits'] > 0:
            hit_reward = 1.0 / max(1, game_state['num_blue_hits'])  # Diminishing returns
            for agent in self.blue_agents:
                rewards[agent] += hit_reward
                
        if game_state['num_red_hits'] > 0:
            hit_reward = 1.0 / max(1, game_state['num_red_hits'])
            for agent in self.red_agents:
                rewards[agent] += hit_reward
        
        self.previous_scores = current_scores
        return rewards
    
    def _check_termination(self, game_state: Dict) -> Tuple[bool, bool]:
        """Check if episode should terminate"""
        # Game ends if max score reached
        terminated = (game_state['score1'] >= self.game.max_score or 
                     game_state['score2'] >= self.game.max_score)
        
        # Episode truncated if max steps reached
        truncated = self.current_step >= self.max_episode_steps
        
        return terminated, truncated
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset game
        self.game.reset_game()
        self.current_step = 0
        self.previous_scores = [0, 0]
        
        # Get initial state
        game_state = self.game.get_game_state()
        
        # Create observations for all agents
        observations = {}
        for agent in self.agents:
            observations[agent] = self._get_observation_for_agent(agent, game_state)
        
        info = {
            'game_state': game_state,
            'step': self.current_step
        }
        
        return observations, info
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict]:
        """Execute one step in the environment"""
        
        # Convert actions to game format
        blue_decisions = []
        red_decisions = []
        
        # Process blue team actions
        for agent in self.blue_agents:
            if agent in actions:
                blue_decisions.append(self._convert_action(actions[agent]))
            else:
                blue_decisions.append({'move_x': 0.0, 'move_y': 0.0})
        
        # Process red team actions  
        for agent in self.red_agents:
            if agent in actions:
                red_decisions.append(self._convert_action(actions[agent]))
            else:
                red_decisions.append({'move_x': 0.0, 'move_y': 0.0})
        
        # Update game
        game_state = self.game.update_one_frame(
            blue_decisions, 
            red_decisions, 
            render=(self.render_mode == 'human')
        )
        
        self.current_step += 1
        
        # Calculate rewards
        rewards = self._calculate_rewards(game_state)
        
        # Check termination
        terminated, truncated = self._check_termination(game_state)
        
        # Create observations
        observations = {}
        for agent in self.agents:
            observations[agent] = self._get_observation_for_agent(agent, game_state)
        
        # Create termination/truncation dicts for all agents
        terminated_dict = {agent: terminated for agent in self.agents}
        truncated_dict = {agent: truncated for agent in self.agents}
        
        info = {
            'game_state': game_state,
            'step': self.current_step,
            'blue_score': game_state['score1'],
            'red_score': game_state['score2']
        }
        
        return observations, rewards, terminated_dict, truncated_dict, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == 'human':
            # Game already renders when update_one_frame is called with render=True
            pygame.display.flip()
            self.game.clock.tick(60)
        elif self.render_mode == 'rgb_array':
            # Capture the pygame surface as RGB array
            if hasattr(self.game, 'screen'):
                # Convert pygame surface to numpy array
                rgb_array = pygame.surfarray.array3d(self.game.screen)
                return rgb_array.transpose((1, 0, 2))  # Correct orientation
        
        return None
    
    def close(self):
        """Close the environment"""
        if hasattr(self.game, 'screen'):
            pygame.quit()


# Wrapper for single agent control (controls one paddle)
class AirHockey2v2SingleAgentEnv(gym.Env):
    """
    Single agent wrapper for 2v2 Air Hockey.
    Agent controls one paddle while others use simple AI or random actions.
    """
    
    def __init__(self, 
                 agent_id: str = 'blue_A',
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = 1000,
                 ai_policy: str = 'random'):
        
        assert agent_id in ['blue_A', 'blue_B', 'red_A', 'red_B'], f"Invalid agent_id: {agent_id}"
        
        self.agent_id = agent_id
        self.ai_policy = ai_policy
        self.env = AirHockey2v2Env(render_mode, max_episode_steps)
        
        # Single agent has same action and observation space as multi-agent
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.metadata = self.env.metadata
    
    def _get_ai_action(self, agent_id: str, game_state: Dict) -> np.ndarray:
        """Get AI action for non-controlled agents"""
        if self.ai_policy == 'random':
            return np.random.uniform(-1.0, 1.0, size=2)
        elif self.ai_policy == 'chase_disc':
            # Simple AI that chases the disc
            if agent_id.startswith('blue'):
                if agent_id == 'blue_A':
                    paddle_x = game_state['blueA_paddle_x']
                    paddle_y = game_state['blueA_paddle_y']
                else:
                    paddle_x = game_state['blueB_paddle_x'] 
                    paddle_y = game_state['blueB_paddle_y']
            else:
                if agent_id == 'red_A':
                    paddle_x = game_state['redA_paddle_x']
                    paddle_y = game_state['redA_paddle_y']
                else:
                    paddle_x = game_state['redB_paddle_x']
                    paddle_y = game_state['redB_paddle_y']
            
            # Move towards disc
            disc_x = game_state['disc_x']
            disc_y = game_state['disc_y']
            
            dx = disc_x - paddle_x
            dy = disc_y - paddle_y
            
            # Normalize movement
            move_x = np.clip(dx * 2.0, -1.0, 1.0)  # Amplify movement
            move_y = np.clip(dy * 2.0, -1.0, 1.0)
            
            return np.array([move_x, move_y])
        
        return np.array([0.0, 0.0])
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment"""
        observations, info = self.env.reset(seed, options)
        return observations[self.agent_id], info
    
    def step(self, action: np.ndarray):
        """Step environment"""
        # Create action dict with AI actions for other agents
        actions = {}
        game_state = self.env.game.get_game_state()
        
        for agent in self.env.agents:
            if agent == self.agent_id:
                actions[agent] = action
            else:
                actions[agent] = self._get_ai_action(agent, game_state)
        
        observations, rewards, terminated, truncated, info = self.env.step(actions)
        
        return (observations[self.agent_id], 
                rewards[self.agent_id],
                terminated[self.agent_id], 
                truncated[self.agent_id],
                info)
    
    def render(self):
        """Render environment"""
        return self.env.render()
    
    def close(self):
        """Close environment"""
        self.env.close()


# Register environments with gymnasium
if __name__ == "__main__":
    # Example usage
    
    # Multi-agent environment
    print("Testing Multi-Agent Environment...")
    env = AirHockey2v2Env(render_mode='human')
    
    obs, info = env.reset()
    print(f"Initial observations keys: {list(obs.keys())}")
    print(f"Observation shape: {obs['blue_A'].shape}")
    
    # Random actions for all agents
    for step in range(100):
        actions = {}
        for agent in env.agents:
            actions[agent] = env.action_space.sample()
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if any(terminated.values()) or any(truncated.values()):
            print(f"Episode finished at step {step}")
            print(f"Final scores: Blue {info['blue_score']}, Red {info['red_score']}")
            break
    
    env.close()
   """
    # Single agent environment
    print("\nTesting Single-Agent Environment...")
    single_env = AirHockey2v2SingleAgentEnv(
        agent_id='blue_A',
        render_mode='human', 
        ai_policy='chase_disc'
    )
    
    obs, info = single_env.reset()
    print(f"Single agent observation shape: {obs.shape}")
    
    for step in range(100):
        action = single_env.action_space.sample()
        obs, reward, terminated, truncated, info = single_env.step(action)
        
        if terminated or truncated:
            print(f"Single agent episode finished at step {step}")
            break
    
    single_env.close()
"""