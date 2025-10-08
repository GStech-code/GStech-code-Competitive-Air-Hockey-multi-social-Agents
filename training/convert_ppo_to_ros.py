#!/usr/bin/env python3
"""
Convert PPO checkpoint to ROS-compatible format
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

# Fix import path - add parent directory to sys.path if running as script
if __name__ == "__main__":
    # Add parent directory to path to enable imports
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

# Now import from training module
try:
    from training.train_ppo import ActorCritic, PPOConfig, AirHockeyEnv
except ImportError:
    # If that fails, try direct import (when running from training directory)
    from train_ppo import ActorCritic, PPOConfig, AirHockeyEnv

class PPOAgentPolicy:
    """Wrapper for PPO policy to work with ROS agent"""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """Initialize policy from checkpoint
        
        Args:
            checkpoint_path: Path to PPO checkpoint
            device: Device to run on (cpu/cuda)
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.network = None
        
        # Load the network from checkpoint
        self._load_from_checkpoint(checkpoint_path)
    
    def _load_from_checkpoint(self, checkpoint_path: str):
        """Load network from checkpoint using actual classes"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        # PyTorch 2.6+ requires weights_only=False for custom classes
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Get the state dict - it's stored as 'model_state_dict' in this checkpoint
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('network_state_dict', checkpoint.get('state_dict', checkpoint)))
        
        # Infer dimensions from the saved network state
        print("Inferring network dimensions from checkpoint...")
        
        # Infer obs_dim from first layer of shared_net
        if 'shared_net.0.weight' in state_dict:
            obs_dim = state_dict['shared_net.0.weight'].shape[1]
            hidden_dim = state_dict['shared_net.0.weight'].shape[0]
        else:
            raise ValueError("Could not infer dimensions from checkpoint")
        
        # Infer act_dim from actor_mean output layer  
        if 'actor_mean.2.weight' in state_dict:
            act_dim = state_dict['actor_mean.2.weight'].shape[0]
        elif 'actor_mean.1.weight' in state_dict:
            act_dim = state_dict['actor_mean.1.weight'].shape[0]
        else:
            raise ValueError("Could not infer action dimension from checkpoint")
        
        print(f"  Inferred: obs_dim={obs_dim}, act_dim={act_dim}, hidden_dim={hidden_dim}")
        
        # Get config from checkpoint or create default
        if 'config' in checkpoint:
            config = checkpoint['config']
            print("  Using config from checkpoint")
        else:
            config = PPOConfig()
            config.hidden_dim = hidden_dim
            print("  Using default config with inferred hidden_dim")
        
        # Define scenario parameters (same as in train_ppo.py main function)
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
        
        # Create actual AirHockeyEnv with config and scenario_params
        env = AirHockeyEnv(config, scenario_params)
        print(f"  Created AirHockeyEnv with obs_dim={env.obs_dim}, action_dim={env.action_dim}")
        
        # Initialize the network with config and env
        self.network = ActorCritic(config, env)
        
        # Load the state dict
        self.network.load_state_dict(state_dict)
        self.network.to(self.device)
        self.network.eval()
        
        print("Successfully loaded network from checkpoint")
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from observation
        
        Args:
            obs: Observation array
            
        Returns:
            Action array
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_mean, _ = self.network(obs_tensor)
            action = action_mean.squeeze(0).cpu().numpy()
            
        return action
    
    def get_actions_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """Get actions for batch of observations
        
        Args:
            obs_batch: Batch of observations [batch_size, obs_dim]
            
        Returns:
            Batch of actions [batch_size, act_dim]
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
            action_means, _ = self.network(obs_tensor)
            actions = action_means.cpu().numpy()
            
        return actions


class ROSAgentConverter:
    """Convert PPO checkpoint to ROS-compatible format"""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """Initialize converter
        
        Args:
            checkpoint_path: Path to PPO checkpoint
            device: Device to run on
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.agent = PPOAgentPolicy(checkpoint_path, device)
    
    def save_ros_format(self, output_dir: str, agent_id: int = 0):
        """Save policy in ROS-compatible format
        
        Args:
            output_dir: Directory to save policy
            agent_id: Agent ID for naming
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the network state dict
        policy_path = output_dir / f"agent_{agent_id}_policy.pt"
        torch.save({
            'state_dict': self.agent.network.state_dict(),
            'network_type': 'ActorCritic',
            'checkpoint_source': self.checkpoint_path,
        }, policy_path)
        
        # Save metadata
        metadata = {
            'agent_id': agent_id,
            'network_type': 'ActorCritic',
            'checkpoint_source': str(self.checkpoint_path),
            'device': str(self.device),
        }
        
        metadata_path = output_dir / f"agent_{agent_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved ROS-compatible policy to:")
        print(f"  Policy: {policy_path}")
        print(f"  Metadata: {metadata_path}")
        
        return policy_path, metadata_path


def convert_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    num_agents: int = 2,
    device: str = "cpu"
):
    """Convert PPO checkpoint to ROS format for multiple agents
    
    Args:
        checkpoint_path: Path to PPO checkpoint
        output_dir: Output directory for ROS policies
        num_agents: Number of agents per team
        device: Device to use
    """
    print(f"Converting checkpoint for {num_agents} agents per team...")
    
    # Create converter
    converter = ROSAgentConverter(checkpoint_path, device)
    
    # Save for each agent
    for agent_id in range(num_agents * 2):  # 2 teams
        team_id = agent_id // num_agents
        team_name = "blue" if team_id == 0 else "red"
        agent_num = agent_id % num_agents
        
        agent_dir = Path(output_dir) / f"team_{team_name}" / f"agent_{agent_num}"
        converter.save_ros_format(agent_dir, agent_id)
    
    print(f"\nConversion complete! Policies saved to {output_dir}")
    print(f"Structure:")
    print(f"  {output_dir}/")
    print(f"    team_blue/")
    for i in range(num_agents):
        print(f"      agent_{i}/")
    print(f"    team_red/")
    for i in range(num_agents):
        print(f"      agent_{i}/")


def main():
    """Main conversion function"""
    parser = argparse.ArgumentParser(description="Convert PPO checkpoint to ROS format")
    
    # Accept checkpoint as both positional and optional argument for compatibility
    parser.add_argument(
        "checkpoint", 
        nargs='?',  # Make positional argument optional
        help="Path to PPO checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_flag",
        help="Path to PPO checkpoint (alternative)"
    )
    parser.add_argument(
        "--output-dir", "--output",  
        dest="output_dir",
        default="policies/trained_ppo",
        help="Output directory for ROS policies"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=2,
        help="Number of agents per team"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Use checkpoint from either positional or flag argument
    checkpoint_path = args.checkpoint or args.checkpoint_flag
    if not checkpoint_path:
        parser.error("checkpoint path is required")
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1
    
    # Convert checkpoint
    try:
        convert_checkpoint(
            checkpoint_path,
            args.output_dir,
            args.num_agents,
            args.device
        )
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())