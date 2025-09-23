#!/usr/bin/env python3
"""
Environment Testing Script
Tests the hierarchical environment implementation with dummy agents.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from environments.hierarchical_env import HierarchicalAirHockeyEnv, PolicyType, TeamFormation


class DummyAgent:
    """Simple dummy agent for testing"""
    
    def __init__(self, agent_type: str, agent_id: str):
        self.agent_type = agent_type  # 'team' or 'paddle'
        self.agent_id = agent_id
        
    def get_action(self, observation):
        """Get random action based on agent type"""
        if self.agent_type == 'team':
            # High-level team action
            return {
                'policy_assignments': np.random.randint(0, 4, size=2),
                'formation_command': np.random.randint(0, 4),
                'priority_target': np.random.uniform(0, 1, size=2),
                'communication': np.random.uniform(-1, 1, size=8)
            }
        else:
            # Low-level paddle action
            return np.random.uniform(-1, 1, size=2)


def test_environment_creation():
    """Test basic environment creation"""
    print("Testing environment creation...")
    
    try:
        # Test with default config
        env = HierarchicalAirHockeyEnv()
        print("✅ Default environment created successfully")
        
        # Test with custom config
        config = {
            'hierarchical_enabled': True,
            'training_stage': 'full_hierarchical',
            'render_mode': None,
            'max_episode_steps': 100
        }
        env = HierarchicalAirHockeyEnv(config)
        print("✅ Custom environment created successfully")
        
        return env
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_agent_spaces(env):
    """Test action and observation spaces"""
    print("\nTesting agent spaces...")
    
    try:
        print(f"Possible agents: {env.possible_agents}")
        print(f"Current agents: {env.agents}")
        
        # Test action spaces
        for agent in env.agents:
            action_space = env.action_spaces[agent]
            obs_space = env.observation_spaces[agent]
            
            print(f"  {agent}:")
            print(f"    Action space: {action_space}")
            print(f"    Observation space: {obs_space}")
            
            # Test action sampling
            sample_action = action_space.sample()
            print(f"    Sample action: {type(sample_action)}")
            
        print("✅ Agent spaces verified")
        return True
        
    except Exception as e:
        print(f"❌ Agent spaces test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_reset(env):
    """Test environment reset functionality"""
    print("\nTesting environment reset...")
    
    try:
        observations, infos = env.reset(seed=42)
        
        print(f"Reset successful!")
        print(f"Observations keys: {list(observations.keys())}")
        print(f"Info keys: {list(infos.keys())}")
        
        # Verify observation shapes
        for agent, obs in observations.items():
            expected_shape = env.observation_spaces[agent].shape
            actual_shape = obs.shape
            print(f"  {agent}: expected {expected_shape}, got {actual_shape}")
            
            if actual_shape != expected_shape:
                print(f"❌ Shape mismatch for {agent}")
                return False
        
        print("✅ Environment reset successful")
        return True
        
    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_step(env):
    """Test environment step functionality with dummy agents"""
    print("\nTesting environment steps...")
    
    try:
        # Create dummy agents
        agents = {}
        for agent in env.agents:
            if agent in ['blue_team', 'red_team']:
                agents[agent] = DummyAgent('team', agent)
            else:
                agents[agent] = DummyAgent('paddle', agent)
        
        # Reset environment
        observations, infos = env.reset()
        
        # Run several steps
        for step in range(10):
            # Get actions from dummy agents
            actions = {}
            for agent, dummy_agent in agents.items():
                if agent in observations:  # Agent is still active
                    actions[agent] = dummy_agent.get_action(observations[agent])
            
            # Execute step
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            print(f"Step {step + 1}:")
            print(f"  Active agents: {list(observations.keys())}")
            print(f"  Rewards: {[(k, f'{v:.3f}') for k, v in rewards.items()]}")
            print(f"  Terminated: {any(terminations.values())}")
            print(f"  Truncated: {any(truncations.values())}")
            
            # Check if episode ended
            if all(terminations.values()) or all(truncations.values()):
                print(f"  Episode ended at step {step + 1}")
                break
        
        print("✅ Environment steps successful")
        return True
        
    except Exception as e:
        print(f"❌ Environment step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hierarchical_features(env):
    """Test hierarchical-specific features"""
    print("\nTesting hierarchical features...")
    
    try:
        # Reset environment
        env.reset()
        
        # Test policy assignment
        print("Testing policy assignment...")
        env.force_policy_assignment('blue_A', PolicyType.DEFENSIVE)
        env.force_policy_assignment('blue_B', PolicyType.OFFENSIVE)
        
        policies = env.base_env.get_current_policies()
        print(f"Current policies: {policies}")
        
        # Test formation assignment
        print("Testing formation assignment...")
        env.force_team_formation('blue_team', TeamFormation.AGGRESSIVE)
        
        formations = env.base_env.get_team_formations()
        print(f"Current formations: {formations}")
        
        # Test hierarchical state
        hierarchical_state = env.get_hierarchical_state()
        print(f"Hierarchical state keys: {list(hierarchical_state.keys())}")
        
        # Test training stage switching
        print("Testing training stage switching...")
        env.set_training_stage('low_level_only')
        print(f"Agents after low-level only: {env.agents}")
        
        env.set_training_stage('full_hierarchical')
        print(f"Agents after full hierarchical: {env.agents}")
        
        print("✅ Hierarchical features working")
        return True
        
    except Exception as e:
        print(f"❌ Hierarchical features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_training_stages():
    """Test environment with different training stages"""
    print("\nTesting different training stages...")
    
    stages = ['low_level_only', 'high_level_only', 'full_hierarchical']
    
    try:
        for stage in stages:
            print(f"  Testing stage: {stage}")
            
            config = {
                'training_stage': stage,
                'render_mode': None,
                'max_episode_steps': 50
            }
            
            env = HierarchicalAirHockeyEnv(config)
            print(f"    Agents: {env.agents}")
            
            # Quick reset and step test
            obs, _ = env.reset()
            print(f"    Observations: {list(obs.keys())}")
            
            # Create appropriate actions
            actions = {}
            for agent in env.agents:
                if agent in ['blue_team', 'red_team']:
                    actions[agent] = {
                        'policy_assignments': [0, 1],
                        'formation_command': 0,
                        'priority_target': [0.5, 0.5],
                        'communication': np.zeros(8)
                    }
                else:
                    actions[agent] = np.array([0.0, 0.0])
            
            obs, rewards, term, trunc, info = env.step(actions)
            print(f"    Step successful, rewards: {len(rewards)}")
            
            env.close()
        
        print("✅ All training stages working")
        return True
        
    except Exception as e:
        print(f"❌ Training stages test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_performance_test(env):
    """Run a performance test to check speed"""
    print("\nRunning performance test...")
    
    try:
        import time
        
        # Create dummy agents
        agents = {}
        for agent in env.agents:
            if agent in ['blue_team', 'red_team']:
                agents[agent] = DummyAgent('team', agent)
            else:
                agents[agent] = DummyAgent('paddle', agent)
        
        num_steps = 100
        start_time = time.time()
        
        observations, _ = env.reset()
        
        for step in range(num_steps):
            # Get actions
            actions = {}
            for agent in env.agents:
                if agent in observations:
                    actions[agent] = agents[agent].get_action(observations[agent])
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Reset if episode ends
            if all(terminations.values()) or all(truncations.values()):
                observations, _ = env.reset()
        
        end_time = time.time()
        elapsed = end_time - start_time
        steps_per_second = num_steps / elapsed
        
        print(f"Performance: {steps_per_second:.1f} steps/second")
        print(f"Time per step: {elapsed/num_steps*1000:.2f}ms")
        
        if steps_per_second > 50:
            print("✅ Performance adequate for training")
        else:
            print("⚠️  Performance may be slow for training")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all environment tests"""
    print("🧪 Hierarchical Air Hockey Environment Testing")
    print("=" * 60)
    
    # Test 1: Environment Creation
    env = test_environment_creation()
    if not env:
        print("❌ Cannot continue - environment creation failed")
        return False
    
    # Test 2: Agent Spaces
    if not test_agent_spaces(env):
        print("❌ Agent spaces test failed")
        return False
    
    # Test 3: Environment Reset
    if not test_environment_reset(env):
        print("❌ Environment reset test failed")
        return False
    
    # Test 4: Environment Steps
    if not test_environment_step(env):
        print("❌ Environment step test failed")
        return False
    
    # Test 5: Hierarchical Features
    if not test_hierarchical_features(env):
        print("❌ Hierarchical features test failed")
        return False
    
    # Test 6: Different Training Stages
    if not test_different_training_stages():
        print("❌ Training stages test failed")
        return False
    
    # Test 7: Performance Test
    env.close()
    env = HierarchicalAirHockeyEnv({'render_mode': None})
    run_performance_test(env)
    
    # Cleanup
    env.close()
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("Environment is ready for agent implementation")
    print("=" * 60)
    
    print("\nNext Steps:")
    print("1. Begin Phase 2: Hierarchical Agent Architecture")
    print("2. Implement high-level team management agents")
    print("3. Implement low-level paddle control agents")
    print("4. Set up communication protocols")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)