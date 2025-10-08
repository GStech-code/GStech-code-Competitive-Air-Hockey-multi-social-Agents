"""
Test Converted ROS Agent Policies

This script tests the actual converted ROS policies (.pkl files) that would be 
used in the ROS environment. It shows pygame visualization and collects metrics.

Usage:
    python test_ros_agent.py --policy-dir policies/trained_ppo --visualize
    python test_ros_agent.py --policy-dir policies/trained_ppo --episodes 10
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import json
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))
from training.convert_ppo_to_ros import PPOAgentPolicy, PPOTeamPolicy

from air_hockey_ros.simulations.base import BaseSimulation


class ROSAgentTester:
    """Test converted ROS agent policies"""
    
    def __init__(self, policy_dir: str, scenario_params: Dict, visualize: bool = False):
        self.policy_dir = Path(policy_dir)
        self.scenario_params = scenario_params
        self.visualize = visualize
        
        # Load agents
        self.agents = self._load_agents()
        self.num_agents = len(self.agents)
        
        # Create simulation
        self.sim = BaseSimulation(view=visualize)
        
    def _load_agents(self) -> List:
        """Load all agent policies from directory"""
        agents = []
        agent_files = sorted(self.policy_dir.glob("ppo_agent_*.pkl"))
        
        if not agent_files:
            raise FileNotFoundError(f"No agent files found in {self.policy_dir}")
        
        print(f"\n📦 Loading agents from {self.policy_dir}:")
        for agent_file in agent_files:
            with open(agent_file, 'rb') as f:
                agent = pickle.load(f)
                agents.append(agent)
                print(f"  ✓ Loaded {agent_file.name}")
        
        return agents
    
    def test_episode(self, opponent_type: str = 'random', max_steps: int = 3600,
                    render: bool = None) -> Dict:
        """Run a single test episode"""
        if render is None:
            render = self.visualize
        
        # Reset simulation
        self.sim.reset_game(
            num_agents_team_a=self.num_agents,
            num_agents_team_b=self.num_agents,
            **self.scenario_params
        )
        
        team_a_score = 0
        team_b_score = 0
        episode_length = 0
        
        for step in range(max_steps):
            world_state = self.sim.get_world_state()
            
            # Get actions from ROS agents (Team A)
            team_a_actions = []
            for i, agent in enumerate(self.agents):
                vx, vy = agent.update(world_state)
                team_a_actions.append((i, vx, vy))
            
            # Generate opponent actions (Team B)
            team_b_actions = self._get_opponent_actions(
                opponent_type, 
                world_state,
                self.num_agents
            )
            
            # Combine all commands
            commands = team_a_actions + team_b_actions
            
            # Step simulation
            self.sim.apply_commands(commands)
            
            # Check for scoring
            new_state = self.sim.get_world_state()
            if new_state['team_a_score'] > team_a_score:
                team_a_score = new_state['team_a_score']
            if new_state['team_b_score'] > team_b_score:
                team_b_score = new_state['team_b_score']
            
            episode_length += 1
            
            # Check if game should end
            if team_a_score >= 5 or team_b_score >= 5:
                break
            
            # Check if visualization window closed
            if render and self.sim._py_view and not self.sim._py_view.pump_events():
                break
        
        winner = 'A' if team_a_score > team_b_score else 'B' if team_b_score > team_a_score else 'Draw'
        
        return {
            'team_a_score': team_a_score,
            'team_b_score': team_b_score,
            'episode_length': episode_length,
            'winner': winner
        }
    
    def _get_opponent_actions(self, opponent_type: str, world_state: Dict, 
                              num_opponents: int) -> List[Tuple]:
        """Generate opponent actions based on type"""
        actions = []
        start_id = self.num_agents
        
        if opponent_type == 'random':
            for i in range(num_opponents):
                vx = np.random.randint(-1, 2)
                vy = np.random.randint(-1, 2)
                actions.append((start_id + i, vx, vy))
        
        elif opponent_type == 'static':
            for i in range(num_opponents):
                actions.append((start_id + i, 0, 0))
        
        elif opponent_type == 'simple':
            # Simple defensive AI: move toward puck
            puck_x = world_state['puck_x']
            puck_y = world_state['puck_y']
            
            for i in range(num_opponents):
                agent_x = world_state['agent_x'][start_id + i]
                agent_y = world_state['agent_y'][start_id + i]
                
                # Move toward puck (simple)
                vx = 0
                vy = 0
                
                if agent_x < puck_x - 20:
                    vx = 1
                elif agent_x > puck_x + 20:
                    vx = -1
                
                if agent_y < puck_y - 20:
                    vy = 1
                elif agent_y > puck_y + 20:
                    vy = -1
                
                actions.append((start_id + i, vx, vy))
        
        else:
            # Default to static
            for i in range(num_opponents):
                actions.append((start_id + i, 0, 0))
        
        return actions
    
    def run_evaluation(self, num_episodes: int = 10, 
                      opponent_type: str = 'random') -> Dict:
        """Run multiple evaluation episodes"""
        results = []
        
        print(f"\n🎮 Running {num_episodes} episodes against '{opponent_type}' opponent...")
        print("=" * 60)
        
        for ep in range(num_episodes):
            # Only visualize first episode
            render = (ep == 0 and self.visualize)
            
            result = self.test_episode(
                opponent_type=opponent_type,
                render=render
            )
            results.append(result)
            
            # Print progress
            print(f"Episode {ep + 1}/{num_episodes}: "
                  f"Score {result['team_a_score']}-{result['team_b_score']}, "
                  f"Winner: {result['winner']}, "
                  f"Length: {result['episode_length']} steps")
        
        # Compute statistics
        stats = self._compute_statistics(results)
        return stats
    
    def _compute_statistics(self, results: List[Dict]) -> Dict:
        """Compute aggregate statistics"""
        stats = {
            'num_episodes': len(results),
            'mean_team_a_score': np.mean([r['team_a_score'] for r in results]),
            'mean_team_b_score': np.mean([r['team_b_score'] for r in results]),
            'mean_episode_length': np.mean([r['episode_length'] for r in results]),
            'win_rate': np.mean([1 if r['winner'] == 'A' else 0 for r in results]),
            'draw_rate': np.mean([1 if r['winner'] == 'Draw' else 0 for r in results]),
            'loss_rate': np.mean([1 if r['winner'] == 'B' else 0 for r in results]),
            'mean_goal_diff': np.mean([r['team_a_score'] - r['team_b_score'] for r in results]),
        }
        return stats
    
    def close(self):
        """Clean up resources"""
        if self.sim and self.sim._py_view:
            self.sim.end_game()


def print_statistics(stats: Dict):
    """Pretty print statistics"""
    print("\n" + "=" * 60)
    print("📊 ROS AGENT EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of Episodes: {stats['num_episodes']}")
    print(f"\nScore Statistics:")
    print(f"  Mean Score (Team A): {stats['mean_team_a_score']:.2f}")
    print(f"  Mean Score (Team B): {stats['mean_team_b_score']:.2f}")
    print(f"  Mean Goal Difference: {stats['mean_goal_diff']:+.2f}")
    print(f"  Mean Episode Length: {stats['mean_episode_length']:.1f} steps")
    print(f"\nWin/Loss Statistics:")
    print(f"  Win Rate (Team A): {stats['win_rate']*100:.1f}%")
    print(f"  Draw Rate: {stats['draw_rate']*100:.1f}%")
    print(f"  Loss Rate (Team B): {stats['loss_rate']*100:.1f}%")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Test converted ROS agent policies')
    parser.add_argument('--policy-dir', default='policies/trained_ppo',
                       help='Directory containing converted policies')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--opponent', default='random',
                       choices=['random', 'static', 'simple'],
                       help='Opponent type')
    parser.add_argument('--visualize', action='store_true',
                       help='Show pygame visualization (first episode only)')
    parser.add_argument('--save-results', help='Save results to JSON file')
    parser.add_argument('--max-steps', type=int, default=3600,
                       help='Maximum steps per episode')
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
    
    # Check if policy directory exists
    if not Path(args.policy_dir).exists():
        print(f"❌ Error: Policy directory not found: {args.policy_dir}")
        print("\nYou need to convert a checkpoint first:")
        print(f"  python training/convert_ppo_to_ros.py --checkpoint <checkpoint.pt> --output {args.policy_dir}")
        print("  OR use quickstart.sh → Option 5")
        sys.exit(1)
    
    try:
        # Create tester
        tester = ROSAgentTester(
            args.policy_dir,
            scenario_params,
            visualize=args.visualize
        )
        
        # Run evaluation
        stats = tester.run_evaluation(
            num_episodes=args.episodes,
            opponent_type=args.opponent
        )
        
        # Print results
        print_statistics(stats)
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"💾 Results saved to {args.save_results}\n")
        
        # Clean up
        tester.close()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()