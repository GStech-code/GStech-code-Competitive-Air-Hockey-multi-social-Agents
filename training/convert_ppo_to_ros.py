#!/usr/bin/env python3
"""
Convert PPO checkpoint to ROS-compatible format
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
import copy

import torch

# Fix import path
if __name__ == "__main__":
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

# Import training modules
from training.train_ppo import ActorCritic, PPOConfig, AirHockeyEnv
from training.ppo_ros_policy import PPOAgentPolicy


def convert_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    num_agents: int = 2,
    device: str = "cpu"
):
    """Convert PPO checkpoint to ROS-compatible format"""
    print(f"\n{'='*60}")
    print(f"Converting PPO Checkpoint to ROS Format")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")
    print(f"Agents per team: {num_agents}")
    print(f"Device: {device}")
    print()
    
    # Load checkpoint
    print("Loading checkpoint...")
    device_obj = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    
    # Get state dict
    state_dict = checkpoint.get('model_state_dict', 
                               checkpoint.get('network_state_dict',
                                            checkpoint.get('state_dict', checkpoint)))
    
    # Get or create config
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("Using config from checkpoint")
    else:
        config = PPOConfig()
        print("Using default PPOConfig")
    
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
    
    # Create environment
    config.num_agents_team_a = num_agents
    config.num_agents_team_b = num_agents
    env = AirHockeyEnv(config, scenario_params)
    
    print(f"Environment: obs_dim={env.obs_dim}, action_dim={env.action_dim}")
    
    # Get network dimensions
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    hidden_dim = config.hidden_dim
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating {num_agents * 2} agent policies...")
    
    # Convert state dict to CPU
    cpu_state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                      for k, v in state_dict.items()}
    
    # Create and save agent policies
    for agent_id in range(num_agents * 2):
        team_agents = num_agents
        opponent_agents = num_agents
        
        policy = PPOAgentPolicy(
            network_state_dict=copy.deepcopy(cpu_state_dict),
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            agent_id=agent_id,
            num_team_agents=team_agents,
            num_opponent_agents=opponent_agents,
            scenario_params=scenario_params,
            device='cpu'
        )
        
        policy_file = output_path / f"ppo_agent_{agent_id}.pkl"
        with open(policy_file, 'wb') as f:
            pickle.dump(policy, f)
        
        print(f"  ✓ Saved: {policy_file.name}")
    
    print(f"\n{'='*60}")
    print(f"✓ Conversion Complete!")
    print(f"{'='*60}")
    print(f"\nCreated {num_agents * 2} policy files")
    print(f"\nTo test: python test_ros_agent.py --policy-dir {output_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Convert PPO checkpoint to ROS format")
    parser.add_argument("checkpoint", nargs='?', help="Path to checkpoint")
    parser.add_argument("--checkpoint", dest="checkpoint_flag", help="Alternative")
    parser.add_argument("--output", "--output-dir", dest="output_dir", 
                       default="policies/trained_ppo")
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint or args.checkpoint_flag
    if not checkpoint_path:
        parser.error("checkpoint path is required")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        return 1
    
    try:
        convert_checkpoint(checkpoint_path, args.output_dir, args.num_agents, args.device)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())