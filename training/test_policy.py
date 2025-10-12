"""
Test and visualize trained PPO policies

This script allows you to:
1. Test trained policies against different opponents
2. Visualize agent behavior
3. Collect performance metrics

Usage:
    # Test against simple opponent
    python test_policy.py --checkpoint checkpoints/ppo_checkpoint_1000.pt --opponent simple
    
    # Test with visualization
    python test_policy.py --checkpoint checkpoints/ppo_checkpoint_1000.pt --visualize
    
    # Run multiple evaluation episodes
    python test_policy.py --checkpoint checkpoints/ppo_checkpoint_1000.pt --episodes 100
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # Go up one level to project root

from air_hockey_ros.simulations.base import BaseSimulation, PygameView
from train_ppo import PPOConfig, ActorCritic, AirHockeyEnv
from air_hockey_ros.policies.simple_policy import SimpleRegionalAgentPolicy


class PolicyTester:
    """Test trained policies in simulation"""
    
    def __init__(self, checkpoint_path: str, scenario_params: Dict, 
                 visualize: bool = False):
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.config = self.checkpoint['config']
        self.device = torch.device('cpu')
        
        # Import PPO components
        from train_ppo import ActorCritic, AirHockeyEnv
        
        # Create environment
        self.env = AirHockeyEnv(self.config, scenario_params)
        
        # Load trained agent
        self.agent = ActorCritic(self.config, self.env).to(self.device)
        self.agent.load_state_dict(self.checkpoint['model_state_dict'])
        self.agent.eval()
        
        # Visualization
        self.visualize = visualize
        if visualize:
            self.view = PygameView()
            self.view.reset(
                num_agents_team_a=self.config.num_agents_team_a,
                num_agents_team_b=self.config.num_agents_team_b,
                **scenario_params
            )

    def _enforce_half_line(self, actions: np.ndarray, world_state: Dict) -> np.ndarray:
        """Enforce half-line constraint for Team A agents"""
        half_line = self.env.engine.width / 2.0
        actions = actions.copy()  # Don't modify original
        
        for i in range(self.config.num_agents_team_a):
            agent_x = world_state['agent_x'][i]
            if agent_x >= half_line:
                # Discretize action
                ax = actions[i, 0]
                if ax > 0.33:
                    ax_discrete = 1
                elif ax < -0.33:
                    ax_discrete = -1
                else:
                    ax_discrete = 0
                
                # Force move left if not already
                if ax_discrete != -1:
                    actions[i, 0] = -1.0
        
        return actions

    def test_episode(self, opponent_type: str = 'random',
                    render: bool = None, max_steps: int = 3600) -> Dict:
        """Run a single test episode"""
        if render is None:
            render = self.visualize
            
        obs = self.env.reset()
        done = False
        step = 0
        
        episode_reward = 0
        episode_length = 0
        team_a_goals = 0
        team_b_goals = 0
        
        prev_scores = {'team_a_score': 0, 'team_b_score': 0}
        
        while not done and step < max_steps:
            # Get trained agent actions (Team A)
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                action, _, _, _ = self.agent.get_action_and_value(obs_tensor)
                action_team_a = action.cpu().numpy()
            
            # Enforce half-line constraint for Team A
            world_state = self.env.engine.get_world_state()
            action_team_a = self._enforce_half_line(action_team_a, world_state)
            
            # Generate opponent actions (Team B)
            action_team_b = self._generate_opponent_actions(opponent_type)
            
            # Combine all actions
            all_actions = np.vstack([action_team_a, action_team_b])
            
            # Step environment
            obs, reward, done_array, info = self.env.step(all_actions)
            done = done_array.any()
            
            # Track metrics
            episode_reward += reward.sum()
            episode_length += 1
            
            if info['scores']['team_a_score'] > prev_scores['team_a_score']:
                team_a_goals += 1
            if info['scores']['team_b_score'] > prev_scores['team_b_score']:
                team_b_goals += 1
            prev_scores = info['scores'].copy()
            
            # Visualization
            if render and self.view:
                state = self.env.engine.get_world_state()
                if not self.view.draw(state):
                    break
                self.view.tick()
            
            step += 1
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'team_a_score': info['scores']['team_a_score'],
            'team_b_score': info['scores']['team_b_score'],
            'team_a_goals': team_a_goals,
            'team_b_goals': team_b_goals,
            'winner': 'A' if info['scores']['team_a_score'] > info['scores']['team_b_score'] else 
                    'B' if info['scores']['team_b_score'] > info['scores']['team_a_score'] else 'Draw'
        }

    def _generate_opponent_actions(self, opponent_type: str) -> np.ndarray:
        """Generate actions for Team B opponents"""
        num_opponents = self.config.num_agents_team_b
        actions = np.zeros((num_opponents, 2), dtype=np.float32)
        
        world_state = self.env.engine.get_world_state()
        
        if opponent_type == 'random':
            # Random actions in [-1, 1]
            for i in range(num_opponents):
                actions[i, 0] = np.random.uniform(-1, 1)
                actions[i, 1] = np.random.uniform(-1, 1)
        
        elif opponent_type == 'static':
            # No movement (already zeros)
            pass
        
        elif opponent_type == 'simple':
            # Use SimpleRegionalAgentPolicy logic (adapted for Team B perspective)
            puck_x = world_state['puck_x']
            puck_y = world_state['puck_y']
            puck_vx = world_state['puck_vx']
            puck_vy = world_state['puck_vy']
            
            # Team B perspective (right side, reversed coordinates)
            width = self.env.engine.width
            height = self.env.engine.height
            paddle_radius = self.env.engine.paddle_radius
            puck_radius = self.env.engine.puck_radius
            unit_speed = self.env.engine.unit_speed_px
            
            # Flip coordinates for Team B (they defend right side)
            puck_x_flipped = width - puck_x
            puck_vx_flipped = -puck_vx
            
            paddle_margin = puck_radius + paddle_radius
            
            for i in range(num_opponents):
                agent_idx = self.config.num_agents_team_a + i
                agent_x_raw = world_state['agent_x'][agent_idx]
                agent_y_raw = world_state['agent_y'][agent_idx]
                
                # Flip agent position for Team B perspective
                agent_x = width - agent_x_raw
                agent_y = agent_y_raw
                
                # Define Team B's defensive region (right half, flipped to look like left)
                x_min = paddle_radius
                x_max = width / 2 - paddle_radius
                y_min = paddle_radius
                y_max = height - paddle_radius
                
                vx = 0
                vy = 0
                
                # Check if in valid region
                valid_x = x_min <= agent_x <= x_max
                valid_y = y_min <= agent_y <= y_max
                
                if not valid_x:
                    vx = 1 if agent_x < x_min else -1
                if not valid_y:
                    vy = 1 if agent_y < y_min else -1
                
                if valid_x and valid_y:
                    # Puck is ahead (in our half)
                    if agent_x <= puck_x_flipped - paddle_margin:
                        # Chase puck behavior (simplified from up_puck_action)
                        dx = puck_x_flipped - agent_x
                        dy = puck_y - agent_y
                        
                        # Normalize to [-1, 1]
                        if abs(dx) > unit_speed:
                            vx = 1 if dx > 0 else -1
                        if abs(dy) > unit_speed:
                            vy = 1 if dy > 0 else -1
                    else:
                        # Puck behind or close (simplified from down_puck_action)
                        dy = puck_y - agent_y
                        
                        if abs(dy) > paddle_margin + unit_speed:
                            vy = 1 if dy > 0 else -1
                            if agent_x > x_min + unit_speed:
                                vx = -1  # Move back
                        else:
                            # Puck at similar y-level, stay defensive
                            if agent_x > x_min + unit_speed:
                                vx = -1
                
                # Flip actions back for actual game coordinates
                actions[i, 0] = -vx  # Flip x direction
                actions[i, 1] = vy   # Y stays same
        
        return actions
    
    def _discretize(self, val: float) -> int:
        """Discretize continuous action"""
        if val > 0.33:
            return 1
        elif val < -0.33:
            return -1
        return 0
    
    def run_evaluation(self, num_episodes: int, opponent_type: str = 'random') -> Dict:
        """Run multiple evaluation episodes"""
        results = []
        
        print(f"\nRunning {num_episodes} evaluation episodes against '{opponent_type}' opponent...")
        
        for ep in range(num_episodes):
            # Generate opponent actions based on type
            if opponent_type == 'random':
                opponent_actions = [(np.random.randint(-1, 2), np.random.randint(-1, 2)) 
                                  for _ in range(self.config.num_agents_team_b)]
            elif opponent_type == 'static':
                opponent_actions = [(0, 0) for _ in range(self.config.num_agents_team_b)]
            else:
                opponent_actions = None
            
            # Run episode
            result = self.test_episode(
                opponent_type=opponent_type,
                render=(ep == 0 and self.visualize)  # Only render first episode
            )
            results.append(result)
            
            # Print progress
            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep + 1}/{num_episodes}: "
                      f"Score {result['team_a_score']}-{result['team_b_score']}, "
                      f"Winner: {result['winner']}")
        
        # Aggregate statistics
        stats = self._compute_statistics(results)
        return stats
    
    def _compute_statistics(self, results: List[Dict]) -> Dict:
        """Compute aggregate statistics"""
        stats = {
            'num_episodes': len(results),
            'mean_reward': np.mean([r['episode_reward'] for r in results]),
            'std_reward': np.std([r['episode_reward'] for r in results]),
            'mean_length': np.mean([r['episode_length'] for r in results]),
            'mean_team_a_score': np.mean([r['team_a_score'] for r in results]),
            'mean_team_b_score': np.mean([r['team_b_score'] for r in results]),
            'win_rate': np.mean([1 if r['winner'] == 'A' else 0 for r in results]),
            'draw_rate': np.mean([1 if r['winner'] == 'Draw' else 0 for r in results]),
            'loss_rate': np.mean([1 if r['winner'] == 'B' else 0 for r in results]),
            'mean_goal_diff': np.mean([r['team_a_score'] - r['team_b_score'] for r in results]),
        }
        return stats
    
    def close(self):
        """Clean up resources"""
        if self.visualize and self.view:
            self.view.close()


def print_statistics(stats: Dict):
    """Pretty print statistics"""
    print("\n" + "="*50)
    print("EVALUATION STATISTICS")
    print("="*50)
    print(f"Number of Episodes: {stats['num_episodes']}")
    print(f"\nReward Statistics:")
    print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean Episode Length: {stats['mean_length']:.1f} steps")
    print(f"\nGame Statistics:")
    print(f"  Mean Score (A vs B): {stats['mean_team_a_score']:.2f} - {stats['mean_team_b_score']:.2f}")
    print(f"  Mean Goal Difference: {stats['mean_goal_diff']:.2f}")
    print(f"\nWin/Loss Statistics:")
    print(f"  Win Rate (Team A): {stats['win_rate']*100:.1f}%")
    print(f"  Draw Rate: {stats['draw_rate']*100:.1f}%")
    print(f"  Loss Rate (Team B): {stats['loss_rate']*100:.1f}%")
    print("="*50 + "\n")


def compare_checkpoints(checkpoint_paths: List[str], scenario_params: Dict, 
                       num_episodes: int = 50):
    """Compare multiple checkpoints"""
    print("\n" + "="*60)
    print("CHECKPOINT COMPARISON")
    print("="*60)
    
    results = []
    
    for path in checkpoint_paths:
        print(f"\nEvaluating: {path}")
        tester = PolicyTester(path, scenario_params, visualize=False)
        stats = tester.run_evaluation(num_episodes, opponent_type='random')
        stats['checkpoint'] = path
        results.append(stats)
        tester.close()
        
        # Quick summary
        print(f"  Win Rate: {stats['win_rate']*100:.1f}% | "
              f"Avg Reward: {stats['mean_reward']:.2f} | "
              f"Avg Score: {stats['mean_team_a_score']:.2f}-{stats['mean_team_b_score']:.2f}")
    
    # Find best checkpoint
    best_idx = np.argmax([r['win_rate'] for r in results])
    print(f"\n🏆 Best Checkpoint: {results[best_idx]['checkpoint']}")
    print(f"   Win Rate: {results[best_idx]['win_rate']*100:.1f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test trained PPO policies')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--opponent', default='random', 
                       choices=['random', 'static', 'simple'],
                       help='Opponent type')
    parser.add_argument('--visualize', action='store_true', help='Visualize episodes')
    parser.add_argument('--compare', nargs='+', help='Compare multiple checkpoints')
    parser.add_argument('--save-results', help='Save results to JSON file')
    args = parser.parse_args()
    
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
    
    if args.compare:
        # Compare multiple checkpoints
        results = compare_checkpoints(args.compare, scenario_params, args.episodes)
        
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.save_results}")
    
    else:
        # Test single checkpoint
        tester = PolicyTester(
            args.checkpoint, 
            scenario_params, 
            visualize=args.visualize
        )
        
        stats = tester.run_evaluation(args.episodes, args.opponent)
        print_statistics(stats)
        
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Results saved to {args.save_results}")
        
        tester.close()


if __name__ == "__main__":
    main()