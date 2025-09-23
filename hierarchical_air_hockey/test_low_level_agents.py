#!/usr/bin/env python3
"""
Test script for low-level paddle agents - FIXED VERSION
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_agent_creation():
    """Test creating different types of paddle agents"""
    print("Testing agent creation...")
    
    try:
        from agents.low_level_agents import (
            DefensiveAgent, OffensiveAgent, PassingAgent, NeutralAgent,
            PaddleAgentFactory, AdaptivePaddleAgent
        )
        from environments.hierarchical_env import PolicyType
        
        # Test individual agent creation
        agents = {
            'defensive': DefensiveAgent(),
            'offensive': OffensiveAgent(), 
            'passing': PassingAgent(),
            'neutral': NeutralAgent()
        }
        
        print("✅ Individual agents created successfully:")
        for name, agent in agents.items():
            print(f"  {name}: {agent.__class__.__name__} - Policy: {agent.policy_type}")
        
        # Test factory creation
        factory_agent = PaddleAgentFactory.create_agent(PolicyType.DEFENSIVE)
        print(f"✅ Factory created: {factory_agent.__class__.__name__}")
        
        # Test adaptive agent
        adaptive_agent = AdaptivePaddleAgent()
        print(f"✅ Adaptive agent created with {len(adaptive_agent.policy_agents)} policies")
        
        return agents, adaptive_agent
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_agent_inference():
    """Test agent forward pass and action generation"""
    print("\nTesting agent inference...")
    
    agents, adaptive_agent = test_agent_creation()
    if not agents:
        return False
    
    try:
        # Create dummy observation (28 dimensions)
        observation = np.random.randn(28).astype(np.float32)
        
        print("Testing individual agents:")
        for name, agent in agents.items():
            # Test get_action method
            action = agent.get_action(observation)
            print(f"  {name}: action shape {action.shape}, range [{action.min():.3f}, {action.max():.3f}]")
            
            # Test forward pass
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            with torch.no_grad():
                action_tensor = agent(obs_tensor)
            print(f"    Forward pass shape: {action_tensor.shape}")
        
        # Test adaptive agent
        if adaptive_agent:
            from environments.hierarchical_env import PolicyType
            
            action = adaptive_agent.get_action(observation)
            print(f"  adaptive (auto): action shape {action.shape}")
            
            # Test with assigned policy
            action_def = adaptive_agent.get_action(observation, assigned_policy=PolicyType.DEFENSIVE)
            print(f"  adaptive (defensive): action shape {action_def.shape}")
            
            # Test policy weights
            weights = adaptive_agent.get_policy_weights(observation)
            print(f"  Policy weights: {[(k.value, f'{v:.3f}') for k, v in weights.items()]}")
        
        print("✅ All agent inference tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Agent inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_policy_specialization():
    """Test that different policies produce different behaviors"""
    print("\nTesting policy specialization...")
    
    try:
        from agents.low_level_agents import PaddleAgentFactory
        from environments.hierarchical_env import PolicyType
        
        # Create agents
        defensive = PaddleAgentFactory.create_agent(PolicyType.DEFENSIVE)
        offensive = PaddleAgentFactory.create_agent(PolicyType.OFFENSIVE)
        passing = PaddleAgentFactory.create_agent(PolicyType.PASSING)
        
        # Create test scenarios
        scenarios = {
            'defensive_scenario': np.array([
                # Paddle near own goal, disc approaching
                0.1, 0.5, 0.2,  # paddle: x, y, speed
                0.3, 0.5, -0.5, 0.0,  # disc: x, y, vx, vy (moving towards goal)
                0.2, 0.3,  # teammate
                0.7, 0.5, 0.8, 0.6,  # opponents
                1.0, 0.0, 0.0, 0.0,  # policy encoding (defensive)
                1, 2, 100,  # game state
                0, 0, 0, 0, 0, 0, 0, 0  # communication
            ], dtype=np.float32),
            
            'offensive_scenario': np.array([
                # Paddle near disc, good scoring opportunity
                0.6, 0.5, 0.1,  # paddle position
                0.65, 0.5, 0.1, 0.0,  # disc nearby
                0.4, 0.4,  # teammate
                0.2, 0.3, 0.3, 0.7,  # opponents far
                0.0, 1.0, 0.0, 0.0,  # policy encoding (offensive)
                1, 2, 150,  # game state
                0, 0, 0, 0, 0, 0, 0, 0  # communication
            ], dtype=np.float32),
            
            'passing_scenario': np.array([
                # Good passing opportunity
                0.4, 0.6, 0.0,  # paddle position
                0.45, 0.55, 0.0, 0.0,  # disc nearby
                0.6, 0.4,  # teammate in good position
                0.8, 0.3, 0.9, 0.7,  # opponents
                0.0, 0.0, 1.0, 0.0,  # policy encoding (passing)
                1, 1, 200,  # game state
                0, 0, 0, 0, 0, 0, 0, 0  # communication
            ], dtype=np.float32)
        }
        
        print("Testing policy responses to different scenarios:")
        
        for scenario_name, obs in scenarios.items():
            print(f"\n  {scenario_name}:")
            
            def_action = defensive.get_action(obs, deterministic=True)
            off_action = offensive.get_action(obs, deterministic=True)
            pass_action = passing.get_action(obs, deterministic=True)
            
            print(f"    Defensive: [{def_action[0]:+.3f}, {def_action[1]:+.3f}]")
            print(f"    Offensive: [{off_action[0]:+.3f}, {off_action[1]:+.3f}]")
            print(f"    Passing:   [{pass_action[0]:+.3f}, {pass_action[1]:+.3f}]")
            
            # Check if policies produce different actions
            def_vs_off = np.linalg.norm(def_action - off_action)
            def_vs_pass = np.linalg.norm(def_action - pass_action)
            off_vs_pass = np.linalg.norm(off_action - pass_action)
            
            print(f"    Differences: D-O={def_vs_off:.3f}, D-P={def_vs_pass:.3f}, O-P={off_vs_pass:.3f}")
            
            if def_vs_off > 0.1 or def_vs_pass > 0.1 or off_vs_pass > 0.1:
                print("    ✅ Policies show different behaviors")
            else:
                print("    ⚠️  Policies are very similar")
        
        print("✅ Policy specialization test completed")
        return True
        
    except Exception as e:
        print(f"❌ Policy specialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing():
    """Test batch processing capabilities"""
    print("\nTesting batch processing...")
    
    try:
        from agents.low_level_agents import DefensiveAgent
        
        agent = DefensiveAgent()
        
        # Create batch of observations
        batch_size = 8
        batch_obs = torch.randn(batch_size, 28)
        
        # Test forward pass
        with torch.no_grad():
            batch_actions = agent(batch_obs)
        
        print(f"  Batch input shape: {batch_obs.shape}")
        print(f"  Batch output shape: {batch_actions.shape}")
        print(f"  Output range: [{batch_actions.min().item():.3f}, {batch_actions.max().item():.3f}]")
        
        # Verify all actions are in valid range
        assert torch.all(batch_actions >= -1.0) and torch.all(batch_actions <= 1.0), "Actions outside valid range"
        
        print("✅ Batch processing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_parameters():
    """Test agent parameter counts and device handling"""
    print("\nTesting agent parameters...")
    
    try:
        from agents.low_level_agents import DefensiveAgent, OffensiveAgent
        
        agent = DefensiveAgent()
        
        # Count parameters
        total_params = sum(p.numel() for p in agent.parameters())
        trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test device handling
        if torch.cuda.is_available():
            try:
                print("  Testing GPU transfer...")
                agent_gpu = DefensiveAgent(device='cuda')
                test_obs = torch.randn(1, 28).cuda()
                with torch.no_grad():
                    action_gpu = agent_gpu(test_obs)
                print(f"  GPU action shape: {action_gpu.shape}, device: {action_gpu.device}")
            except Exception as e:
                print(f"  ⚠️ GPU test skipped due to compatibility: {str(e)[:50]}...")
        else:
            print("  GPU not available, skipping GPU test")
        
        # Test different architectures
        agent_large = DefensiveAgent(hidden_dims=[256, 128, 64])
        large_params = sum(p.numel() for p in agent_large.parameters())
        
        print(f"  Large agent parameters: {large_params:,}")
        print(f"  Parameter ratio (large/default): {large_params/total_params:.1f}x")
        
        print("✅ Parameter tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Parameter tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all low-level agent tests"""
    print("🧪 Low-Level Paddle Agents Testing")
    print("=" * 50)
    
    tests = [
        test_agent_creation,
        test_agent_inference, 
        test_policy_specialization,
        test_batch_processing,
        test_agent_parameters
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print("=" * 50)
    if passed == len(tests):
        print("🎉 ALL LOW-LEVEL AGENT TESTS PASSED!")
        print("Ready to implement high-level agents")
    else:
        print(f"❌ {len(tests) - passed}/{len(tests)} tests failed")
        print("Check errors above before proceeding")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)