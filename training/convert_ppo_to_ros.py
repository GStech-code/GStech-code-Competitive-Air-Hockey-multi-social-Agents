"""
Convert trained PPO model to ROS-compatible policy format

This script extracts the trained actor network from PPO checkpoints
and creates pickle files compatible with your ROS agent nodes.

Usage:
    python convert_ppo_to_ros.py --checkpoint checkpoints/ppo_checkpoint_1000.pt \
                                  --output policies/trained_ppo.pkl \
                                  --num-agents 2
"""

import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import List

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # Go up one level to project root

from air_hockey_ros import AgentPolicy, TeamPolicy, register_policy
from air_hockey_ros.policies.neural_network_policy.multiagent_paddle_net import MultiAgentPaddleNet


class PPOAgentPolicy(AgentPolicy):
    """ROS-compatible policy using trained PPO actor"""
    
    def __init__(self, agent_id: int, ppo_actor_state: dict, 
                 width: float, height: float, max_speed: float,
                 teammate_ids: List[int], opponent_ids: List[int],
                 device: str = "cpu"):
        super().__init__(agent_id)
        
        # Store configuration
        self.agent_id = int(agent_id)
        self.inv_w = 1.0 / float(width)
        self.inv_h = 1.0 / float(height)
        self.inv_v = 1.0 / float(max_speed)
        self.teammate_ids = list(teammate_ids)
        self.opponent_ids = list(opponent_ids)
        self.device = torch.device(device)
        
        # Create actor network from PPO checkpoint
        self.actor_net = self._create_actor_from_ppo(ppo_actor_state)
        self.actor_net.to(self.device)
        self.actor_net.eval()
        
    def _create_actor_from_ppo(self, state_dict: dict):
        """Extract actor network from PPO checkpoint"""
        import torch.nn as nn
        
        # Reconstruct actor from state dict
        # Assuming PPO used similar architecture
        obs_dim = self._get_obs_dim()
        hidden_dim = 128  # Match PPO config
        
        actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 2),
        )
        
        # Load weights (filter only actor weights)
        actor_weights = {
            k.replace('shared_net.', '').replace('actor_mean.', ''): v
            for k, v in state_dict.items()
            if 'shared_net' in k or 'actor_mean' in k
        }
        
        # Map to sequential layers
        layer_mapping = {
            '0.weight': 'shared_net.0.weight',
            '0.bias': 'shared_net.0.bias',
            '1.weight': 'shared_net.2.weight',
            '1.bias': 'shared_net.2.bias',
            '2.weight': 'actor_mean.0.weight',
            '2.bias': 'actor_mean.0.bias',
            '3.weight': 'actor_mean.2.weight',
            '3.bias': 'actor_mean.2.bias',
        }
        
        new_state_dict = {}
        for new_key, old_key in layer_mapping.items():
            if old_key in state_dict:
                new_state_dict[new_key] = state_dict[old_key]
        
        actor.load_state_dict(new_state_dict, strict=False)
        return actor
        
    def _get_obs_dim(self):
        """Calculate observation dimension"""
        num_teammates = len(self.teammate_ids)
        num_opponents = len(self.opponent_ids)
        return 2 + 4 + num_teammates * 5 + num_opponents * 5
    
    def update(self, world_state: dict) -> tuple[int, int]:
        """Get action from trained policy"""
        # Build observation
        obs = self._build_observation(world_state)
        
        # Get action from network
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actor_net(obs_tensor)
            action = torch.tanh(action)  # Ensure in [-1, 1]
            action = action.squeeze(0).cpu().numpy()
        
        # Discretize to {-1, 0, 1}
        return self._discretize(action[0]), self._discretize(action[1])
    
    def _discretize(self, val: float, threshold: float = 0.33) -> int:
        """Convert continuous action to discrete"""
        if val > threshold:
            return 1
        elif val < -threshold:
            return -1
        return 0
    
    def _build_observation(self, ws: dict) -> np.ndarray:
        """Build observation vector from world state"""
        # Normalize positions
        ax = [x * self.inv_w for x in ws["agent_x"]]
        ay = [y * self.inv_h for y in ws["agent_y"]]
        px = ws["puck_x"] * self.inv_w
        py = ws["puck_y"] * self.inv_h
        pvx = ws["puck_vx"] * self.inv_v
        pvy = ws["puck_vy"] * self.inv_v
        
        obs = []
        
        # Self position
        obs.extend([ax[self.agent_id], ay[self.agent_id]])
        
        # Puck state
        obs.extend([px, py, pvx, pvy])
        
        # Teammate features
        for j in self.teammate_ids:
            obs.extend([
                ax[j] - ax[self.agent_id],  # dx_self
                ay[j] - ay[self.agent_id],  # dy_self
                ax[j] - px,                 # dx_puck
                ay[j] - py,                 # dy_puck
                1.0                         # is_teammate
            ])
        
        # Opponent features
        for j in self.opponent_ids:
            obs.extend([
                ax[j] - ax[self.agent_id],  # dx_self
                ay[j] - ay[self.agent_id],  # dy_self
                ax[j] - px,                 # dx_puck
                ay[j] - py,                 # dy_puck
                0.0                         # is_opponent
            ])
        
        return np.array(obs, dtype=np.float32)


@register_policy('trained_ppo')
class PPOTeamPolicy(TeamPolicy):
    """Team policy using trained PPO agents"""
    
    def __init__(self, ppo_checkpoint_path: str = None, **params):
        super().__init__(**params)
        
        # Load PPO checkpoint
        if ppo_checkpoint_path and Path(ppo_checkpoint_path).exists():
            checkpoint = torch.load(ppo_checkpoint_path, map_location='cpu', weights_only=False)
            self.ppo_state = checkpoint['model_state_dict']
        else:
            raise ValueError(f"PPO checkpoint not found: {ppo_checkpoint_path}")
        
        # Configuration
        self.width = params['width']
        self.height = params['height']
        puck_max_speed = float(params.get('puck_max_speed', 6.0))
        unit_speed_px = float(params.get('unit_speed_px', 4.0))
        self.max_speed = max(puck_max_speed, unit_speed_px) * 1.05
        
        # Device
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Build team/opponent ID lists
        self.teammates_ids = [
            [id for id in self.agents_ids if id != current_agent]
            for current_agent in self.agents_ids
        ]
        
        if self.team == 'A':
            self.opponents_ids = list(range(
                self.num_agents_team_a,
                self.num_agents_team_a + self.num_agents_team_b
            ))
        else:
            self.opponents_ids = list(range(self.num_agents_team_a))
    
    def get_policies(self) -> List[AgentPolicy]:
        """Create PPO agent policies for each agent"""
        return [
            PPOAgentPolicy(
                agent_id=agent_id,
                ppo_actor_state=self.ppo_state,
                width=self.width,
                height=self.height,
                max_speed=self.max_speed,
                teammate_ids=teammates,
                opponent_ids=self.opponents_ids,
                device=self.device
            )
            for agent_id, teammates in zip(self.agents_ids, self.teammates_ids)
        ]


def convert_checkpoint_to_ros(checkpoint_path: str, output_path: str, 
                              num_agents: int, scenario_params: dict):
    """
    Convert PPO checkpoint to ROS-compatible pickled policies
    
    Args:
        checkpoint_path: Path to PPO checkpoint (.pt file)
        output_path: Output path for pickled policy
        num_agents: Number of agents
        scenario_params: Scenario configuration dict
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create team policy with checkpoint
    team_policy = PPOTeamPolicy(
        ppo_checkpoint_path=checkpoint_path,
        num_agents_team_a=num_agents,
        num_agents_team_b=num_agents,
        team='A',
        **scenario_params
    )
    
    # Get individual policies
    policies = team_policy.get_policies()
    
    # Save each policy
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, policy in enumerate(policies):
        policy_path = output_dir / f"ppo_agent_{i}.pkl"
        with open(policy_path, 'wb') as f:
            pickle.dump(policy, f)
        print(f"Saved policy for agent {i} to {policy_path}")
    
    print(f"\nConversion complete! {len(policies)} policies saved.")
    print(f"Use these policies in your launch file:")
    print(f"  team_a_name: 'trained_ppo'")
    print(f"  Make sure to set ppo_checkpoint_path parameter")


def main():
    parser = argparse.ArgumentParser(description='Convert PPO checkpoint to ROS policies')
    parser.add_argument('--checkpoint', required=True, help='Path to PPO checkpoint')
    parser.add_argument('--output', required=True, help='Output directory for policies')
    parser.add_argument('--num-agents', type=int, default=2, help='Number of agents per team')
    parser.add_argument('--width', type=int, default=800, help='Table width')
    parser.add_argument('--height', type=int, default=600, help='Table height')
    parser.add_argument('--puck-max-speed', type=float, default=6.0, help='Puck max speed')
    parser.add_argument('--unit-speed', type=float, default=4.0, help='Unit speed px')
    args = parser.parse_args()
    
    scenario_params = {
        'width': args.width,
        'height': args.height,
        'puck_max_speed': args.puck_max_speed,
        'unit_speed_px': args.unit_speed,
        'goal_gap': 240,
        'goal_offset': 40,
        'paddle_radius': 20,
        'puck_radius': 12,
    }
    
    convert_checkpoint_to_ros(
        args.checkpoint,
        args.output,
        args.num_agents,
        scenario_params
    )


if __name__ == "__main__":
    main()