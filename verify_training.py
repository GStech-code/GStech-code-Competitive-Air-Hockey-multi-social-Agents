#!/usr/bin/env python3
"""
Verification script to diagnose training issues

Usage:
    python verify_training.py checkpoints/curriculum_phase_4_full_game.pt
    python verify_training.py checkpoints/curriculum_phase_4_full_game.pt logs/curriculum_training.log
"""

import torch
import numpy as np
import sys
from pathlib import Path

def verify_checkpoint(checkpoint_path):
    """Verify checkpoint for common issues"""
    print("=" * 60)
    print("CHECKPOINT VERIFICATION")
    print("=" * 60)
    print(f"File: {checkpoint_path}\n")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✓ Checkpoint loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return False
    
    issues = []
    
    # Check 1: Checkpoint structure
    print("1. CHECKPOINT STRUCTURE")
    required_keys = ['model_state_dict', 'config']
    for key in required_keys:
        if key in checkpoint:
            print(f"   ✓ {key}")
        else:
            print(f"   ✗ Missing {key}")
            issues.append(f"Missing {key}")
    
    if 'phase' in checkpoint:
        print(f"   Phase: {checkpoint['phase']}")
    if 'name' in checkpoint:
        print(f"   Name: {checkpoint['name']}")
    print()
    
    # Check 2: Reward scale (if available)
    print("2. REWARD SCALE CHECK")
    if 'rollout_info' in checkpoint:
        rewards = checkpoint['rollout_info'].get('episode_rewards', [])
        if rewards:
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            min_reward = np.min(rewards)
            max_reward = np.max(rewards)
            
            print(f"   Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"   Range: [{min_reward:.2f}, {max_reward:.2f}]")
            
            if abs(mean_reward) > 1e6:
                print("   ✗ CRITICAL: Reward magnitude extremely high!")
                print("   → Apply reward normalization")
                issues.append("Reward explosion")
            elif abs(mean_reward) > 10000:
                print("   ⚠  WARNING: Reward magnitude very high")
                print("   → Consider reward scaling")
            else:
                print("   ✓ Reward scale looks reasonable")
        else:
            print("   - No reward data")
    else:
        print("   - No rollout info in checkpoint")
    print()
    
    # Check 3: Value function scale
    print("3. VALUE FUNCTION CHECK")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        # Find value head weights
        value_weights = None
        for key in state_dict.keys():
            if 'critic' in key.lower() and 'weight' in key:
                if '2' in key or 'linear' in key.lower():
                    value_weights = state_dict[key]
                    break
        
        if value_weights is not None:
            weight_norm = torch.norm(value_weights).item()
            weight_max = torch.max(torch.abs(value_weights)).item()
            weight_mean = torch.mean(torch.abs(value_weights)).item()
            
            print(f"   Weight norm: {weight_norm:.4f}")
            print(f"   Weight max: {weight_max:.4f}")
            print(f"   Weight mean: {weight_mean:.4f}")
            
            if weight_norm > 100:
                print("   ✗ CRITICAL: Value weights exploded!")
                print("   → Reset value head at phase transitions")
                issues.append("Value weight explosion")
            elif weight_norm > 10:
                print("   ⚠  WARNING: Value weights are large")
            else:
                print("   ✓ Value weights look normal")
        else:
            print("   - Could not find value head weights")
    print()
    
    # Check 4: Policy entropy (if available)
    print("4. POLICY ENTROPY CHECK")
    if 'update_info' in checkpoint:
        entropy = checkpoint['update_info'].get('entropy', None)
        if entropy is not None:
            print(f"   Entropy: {entropy:.4f}")
            
            if entropy < 0.1:
                print("   ✗ CRITICAL: Entropy collapsed - no exploration!")
                print("   → Increase ent_coef or reset policy")
                issues.append("Entropy collapse")
            elif entropy < 1.0:
                print("   ⚠  WARNING: Low entropy")
            else:
                print("   ✓ Entropy is healthy")
        else:
            print("   - No entropy data")
    print()
    
    # Check 5: Training progress (if available)
    print("5. TRAINING PROGRESS CHECK")
    if 'rollout_info' in checkpoint:
        goals_for = checkpoint['rollout_info'].get('goals_for', [0])
        goals_against = checkpoint['rollout_info'].get('goals_against', [0])
        
        if goals_for and len(goals_for) > 0:
            mean_goals_for = np.mean(goals_for)
            mean_goals_against = np.mean(goals_against)
            
            print(f"   Goals per episode: {mean_goals_for:.2f} - {mean_goals_against:.2f}")
            
            if mean_goals_for == 0 and mean_goals_against == 0:
                print("   ✗ CRITICAL: No goals scored by anyone!")
                print("   → Agents not hitting puck - check dense rewards")
                issues.append("No goals")
            else:
                print("   ✓ Agents are scoring")
    print()
    
    # Check 6: Loss values (if available)
    print("6. LOSS VALUES CHECK")
    if 'update_info' in checkpoint:
        update_info = checkpoint['update_info']
        
        pg_loss = update_info.get('policy_loss', None)
        v_loss = update_info.get('value_loss', None)
        
        if pg_loss is not None:
            print(f"   Policy loss: {pg_loss:.4f}")
        if v_loss is not None:
            print(f"   Value loss: {v_loss:.4f}")
        
        if v_loss is not None:
            if v_loss > 10000:
                print("   ✗ CRITICAL: Value loss exploded!")
                print("   → Apply reward normalization and value clipping")
                issues.append("Value loss explosion")
            elif v_loss > 100:
                print("   ⚠  WARNING: Value loss is high")
            else:
                print("   ✓ Losses look reasonable")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if issues:
        print("✗ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("1. Apply fixes from the updated training code")
        print("2. Start fresh from Phase 1")
        print("3. Monitor training closely")
    else:
        print("✓ No critical issues detected")
        print("Training appears to be progressing normally")
    
    print("=" * 60)
    return len(issues) == 0


def verify_log_file(log_path):
    """Analyze training log for patterns"""
    print("\n" + "=" * 60)
    print("LOG FILE ANALYSIS")
    print("=" * 60)
    print(f"File: {log_path}\n")
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        print(f"✓ Log file loaded ({len(lines)} lines)\n")
    except FileNotFoundError:
        print(f"✗ Log file not found: {log_path}")
        return False
    
    # Parse metrics
    rewards = []
    value_losses = []
    entropies = []
    goals_for = []
    goals_against = []
    
    for line in lines:
        try:
            if 'Reward:' in line:
                reward_str = line.split('Reward: ')[1].split(' |')[0]
                rewards.append(float(reward_str))
            
            if 'V Loss:' in line:
                vloss_str = line.split('V Loss: ')[1].split(' |')[0]
                value_losses.append(float(vloss_str))
            
            if 'Entropy:' in line:
                entropy_str = line.split('Entropy: ')[1].strip().split('\n')[0]
                entropies.append(float(entropy_str))
            
            if 'Goals:' in line:
                goals_str = line.split('Goals: ')[1].split(' |')[0]
                if '-' in goals_str:
                    gf, ga = goals_str.split('-')
                    goals_for.append(float(gf))
                    goals_against.append(float(ga))
        except:
            continue
    
    # Analyze rewards
    if rewards:
        print("REWARD TREND")
        print(f"  Early (first 10): {np.mean(rewards[:10]):.2f}")
        if len(rewards) > 10:
            print(f"  Recent (last 10): {np.mean(rewards[-10:]):.2f}")
            
            early_mean = np.mean(rewards[:10])
            late_mean = np.mean(rewards[-10:])
            improvement = late_mean - early_mean
            
            print(f"  Improvement: {improvement:+.2f}")
            
            if improvement > 0:
                print("  ✓ Rewards improving")
            elif improvement < -100:
                print("  ✗ Rewards getting worse!")
            else:
                print("  ⚠  Rewards stagnant")
        
        # Check for explosions
        if max([abs(r) for r in rewards]) > 1e6:
            print("  ✗ CRITICAL: Reward explosion detected!")
        print()
    
    # Analyze value losses
    if value_losses:
        print("VALUE LOSS TREND")
        print(f"  Early (first 10): {np.mean(value_losses[:10]):.4f}")
        if len(value_losses) > 10:
            print(f"  Recent (last 10): {np.mean(value_losses[-10:]):.4f}")
        
        if max(value_losses) > 1e6:
            print("  ✗ CRITICAL: Value loss exploded!")
        elif max(value_losses) > 100:
            print("  ⚠  WARNING: Value loss got very high")
        else:
            print("  ✓ Value loss stable")
        print()
    
    # Analyze entropy
    if entropies:
        print("ENTROPY TREND")
        print(f"  Early (first 10): {np.mean(entropies[:10]):.4f}")
        if len(entropies) > 10:
            print(f"  Recent (last 10): {np.mean(entropies[-10:]):.4f}")
        
        if entropies[-1] < 0.1:
            print("  ✗ CRITICAL: Entropy collapsed!")
        elif entropies[-1] < 1.0:
            print("  ⚠  WARNING: Entropy low")
        else:
            print("  ✓ Entropy healthy")
        print()
    
    # Analyze goals
    if goals_for and goals_against:
        print("GOAL SCORING TREND")
        print(f"  Team A goals/ep: {np.mean(goals_for):.2f}")
        print(f"  Team B goals/ep: {np.mean(goals_against):.2f}")
        
        if np.mean(goals_for) == 0:
            print("  ✗ Team A not scoring!")
        else:
            print("  ✓ Team A scoring regularly")
        print()
    
    print("=" * 60)
    return True


def print_usage():
    print("Usage:")
    print("  python verify_training.py <checkpoint_path>")
    print("  python verify_training.py <checkpoint_path> <log_path>")
    print()
    print("Examples:")
    print("  python verify_training.py checkpoints/curriculum_phase_4_full_game.pt")
    print("  python verify_training.py checkpoints/curriculum_phase_1_learn_hitting.pt logs/curriculum_training.log")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    if not Path(checkpoint_path).exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Verify checkpoint
    success = verify_checkpoint(checkpoint_path)
    
    # Verify log if provided
    if len(sys.argv) > 2:
        log_path = sys.argv[2]
        if Path(log_path).exists():
            verify_log_file(log_path)
        else:
            print(f"\n⚠️  Log file not found: {log_path}")
    else:
        # Try to find log automatically
        log_path = "logs/curriculum_training.log"
        if Path(log_path).exists():
            verify_log_file(log_path)
    
    print("\n✓ Verification complete")
    
    if not success:
        print("\n⚠️  Issues detected - see recommendations above")
        sys.exit(1)
