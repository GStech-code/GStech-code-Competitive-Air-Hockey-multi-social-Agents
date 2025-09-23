#!/usr/bin/env python3
"""
Hierarchical Air Hockey Project Setup Script
Creates the complete directory structure and initial configuration files.
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the complete project directory structure"""
    
    directories = [
        # Main source directories
        "src/environments",
        "src/environments/game_core", 
        "src/agents",
        "src/training",
        "src/evaluation",
        "src/utils",
        
        # Checkpoint directories with hierarchical structure
        "checkpoints/high_level/blue_team",
        "checkpoints/high_level/red_team",
        "checkpoints/low_level/defensive",
        "checkpoints/low_level/offensive", 
        "checkpoints/low_level/passing",
        "checkpoints/combined",
        "checkpoints/best_models",
        
        # Configuration and results
        "configs",
        "logs/training",
        "logs/evaluation",
        "logs/tensorboard",
        "results/individual_policies",
        "results/team_performance",
        "results/visualizations",
        
        # Testing and analysis
        "tests/unit",
        "tests/integration",
        "analysis",
        
        # Documentation and examples
        "docs",
        "examples"
    ]
    
    print("Creating hierarchical air hockey project structure...")
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory.startswith("src/"):
            init_file = Path(directory) / "__init__.py"
            init_file.write_text("# Hierarchical Air Hockey Module\n")
            
        print(f"✅ Created: {directory}")
    
    print(f"\n✅ Created {len(directories)} directories")

"""def copy_existing_game_files():
    #Copy existing game files to the new structure
    
    game_files = [
        "game2v2.py",
        "disc.py", 
        "paddle.py",
        "helper_functions.py",
        "air_hockey_gym_env.py"
    ]
    
    target_dir = Path("src/environments/game_core")
    
    print("\nCopying existing game files...")
    
    for file in game_files:
        if Path(file).exists():
            shutil.copy2(file, target_dir / file)
            print(f"✅ Copied: {file}")
        else:
            print(f"⚠️  Not found: {file} (will need to be created)")
    
    # Also check in air_hockey_python directory
    if Path("air_hockey_python").exists():
        for file in game_files:
            source_path = Path("air_hockey_python") / file
            if source_path.exists() and not (target_dir / file).exists():
                shutil.copy2(source_path, target_dir / file)
                print(f"✅ Copied from air_hockey_python/: {file}")
"""
def create_requirements_file():
    """Create requirements.txt with all necessary dependencies"""
    
    requirements_content = """# Hierarchical Air Hockey Requirements
# Core RL Framework
ray[rllib]==2.8.0
torch>=1.13.0,<2.1.0
tensorflow>=2.8.0,<2.15.0

# Multi-Agent RL
pettingzoo>=1.22.0
gymnasium>=0.29.0
stable-baselines3>=2.0.0

# Game Environment
pygame>=2.5.0
PyOpenGL>=3.1.7
numpy>=1.21.0,<1.25.0

# Configuration and Utilities  
pyyaml>=6.0
omegaconf>=2.3.0
hydra-core>=1.3.0

# Monitoring and Visualization
tensorboard>=2.8.0
wandb>=0.15.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Development and Testing
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=5.0.0

# Data handling
pandas>=1.5.0
scipy>=1.9.0

# Optional: For advanced visualization
opencv-python>=4.5.0
pillow>=9.0.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ Created requirements.txt")

def create_config_files():
    """Create initial configuration files"""
    
    # Main training configuration
    training_config = """# Hierarchical Air Hockey Training Configuration
training:
  algorithm: "MADDPG"
  total_timesteps: 10_000_000
  batch_size: 512
  learning_rate: 0.0001
  gamma: 0.99
  tau: 0.005
  
  # Checkpointing
  save_frequency: 100000  # timesteps
  evaluation_frequency: 50000
  keep_checkpoints: 10

hierarchical:
  enabled: true
  levels: 2
  high_level_frequency: 10  # steps between high-level decisions
  low_level_frequency: 1   # continuous low-level control
  
  # Communication settings
  communication_dim: 8
  message_history_length: 5

curriculum:
  enabled: true
  auto_advance: true
  
  stages:
    - name: "individual_policies"
      duration: 1_000_000
      focus: "low_level_only"
      success_threshold: 0.7
      
    - name: "fixed_combinations" 
      duration: 2_000_000
      focus: "policy_combinations"
      success_threshold: 0.75
      
    - name: "adaptive_strategies"
      duration: 5_000_000
      focus: "full_hierarchy"
      success_threshold: 0.8

policies:
  defensive:
    reward_weights:
      goal_protection: 2.0
      interception: 1.5
      positioning: 1.0
      
  offensive: 
    reward_weights:
      scoring: 3.0
      disc_control: 2.0
      pressure: 1.0
      
  passing:
    reward_weights:
      teammate_assist: 2.5
      positioning: 1.5
      coordination: 2.0

rewards:
  high_level:
    team_coordination: 10.0
    strategic_positioning: 5.0
    adaptive_bonus: 15.0
    formation_maintenance: 8.0
    
  low_level:
    policy_adherence: 3.0
    individual_skill: 5.0  
    team_support: 7.0
    execution_quality: 4.0
    
  shared:
    goal_scored: 100.0
    goal_conceded: -50.0
    disc_hit: 5.0
    win_bonus: 200.0
"""
    
    with open("configs/training_config.yaml", "w") as f:
        f.write(training_config)
    
    # Environment configuration
    env_config = """# Environment Configuration
environment:
  name: "HierarchicalAirHockey2v2"
  
  # Game settings
  screen_width: 800
  screen_height: 600
  max_score: 5
  max_episode_steps: 3600
  
  # Physics
  disc_max_speed: 8.0
  paddle_max_speed: 5.0
  friction: 0.98
  restitution: 0.8
  
  # Rendering
  render_training: false
  render_evaluation: true
  fps: 60
  
  # Observations
  observation_type: "hierarchical"  # or "flat"
  normalize_observations: true
  include_history: true
  history_length: 5
  
  # Actions  
  action_type: "continuous"  # or "discrete"
  action_space_size: 2  # [x_vel, y_vel]
  
agents:
  high_level:
    observation_space:
      team_formation: 8
      game_state: 12
      opponent_analysis: 6
      strategic_features: 10
      
    action_space:
      policy_assignments: 4  # 2 per team member
      formation_commands: 3
      priority_targets: 2
      
  low_level:
    observation_space:
      local_state: 15
      assigned_policy: 3
      communication: 8
      
    action_space:
      movement: 2
      execution_intensity: 1
"""
    
    with open("configs/environment_config.yaml", "w") as f:
        f.write(env_config)
    
    print("✅ Created configuration files")

def create_initial_python_files():
    """Create initial Python files with basic structure"""
    
    # Main package init
    main_init = '''"""
Hierarchical Air Hockey Multi-Agent Reinforcement Learning
A sophisticated 2v2 air hockey environment with hierarchical agent control.
"""

__version__ = "0.1.0"
__author__ = "Hierarchical Air Hockey Team"

from .environments import HierarchicalAirHockeyEnv
from .agents import HierarchicalAgent, HighLevelAgent, LowLevelAgent
from .training import HierarchicalTrainer

__all__ = [
    "HierarchicalAirHockeyEnv",
    "HierarchicalAgent", 
    "HighLevelAgent",
    "LowLevelAgent",
    "HierarchicalTrainer"
]
'''
    
    with open("src/__init__.py", "w") as f:
        f.write(main_init)
    
    # Create placeholder files for main modules
    files_to_create = {
        "src/environments/base_env.py": "# Base environment implementation",
        "src/environments/hierarchical_env.py": "# Hierarchical environment wrapper", 
        "src/agents/high_level_agent.py": "# High-level strategic agents",
        "src/agents/low_level_agents.py": "# Low-level paddle control agents",
        "src/agents/hierarchical_policies.py": "# Policy definitions and management",
        "src/training/hierarchical_trainer.py": "# Main training orchestrator",
        "src/training/curriculum_learning.py": "# Curriculum learning implementation",
        "src/training/reward_systems.py": "# Hierarchical reward systems",
        "src/evaluation/evaluator.py": "# Model evaluation framework",
        "src/evaluation/visualization.py": "# Training and performance visualization",
        "src/utils/config.py": "# Configuration management utilities",
        "src/utils/logging_utils.py": "# Logging and monitoring utilities"
    }
    
    for file_path, content in files_to_create.items():
        with open(file_path, "w") as f:
            f.write(content + "\n")
    
    print("✅ Created initial Python module files")

def create_setup_verification():
    """Create a script to verify the setup"""
    
    verification_script = '''#!/usr/bin/env python3
"""
Setup Verification Script
Checks that all components are properly installed and configured.
"""

import sys
import importlib
from pathlib import Path

def check_dependencies():
    """Check if all required packages are available"""
    
    required_packages = [
        "ray", "torch", "gymnasium", "pettingzoo", 
        "pygame", "numpy", "yaml", "matplotlib"
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - NOT FOUND")
    
    if missing:
        print(f"\\n⚠️  Missing packages: {missing}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\\n✅ All required packages found!")
    return True

def check_directory_structure():
    """Verify directory structure is complete"""
    
    required_dirs = [
        "src/environments", "src/agents", "src/training",
        "checkpoints/high_level", "checkpoints/low_level",
        "configs", "logs"
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)
            print(f"❌ {dir_path}")
        else:
            print(f"✅ {dir_path}")
    
    if missing:
        print(f"\\n⚠️  Missing directories: {missing}")
        return False
        
    print("\\n✅ Directory structure complete!")
    return True

def check_config_files():
    """Verify configuration files exist"""
    
    config_files = [
        "configs/training_config.yaml",
        "configs/environment_config.yaml"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file}")
            return False
    
    print("\\n✅ Configuration files complete!")
    return True

def main():
    """Run complete verification"""
    print("🔍 Verifying Hierarchical Air Hockey Setup")
    print("=" * 50)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure), 
        ("Configuration Files", check_config_files)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\\nChecking {name}...")
        if not check_func():
            all_passed = False
    
    print("\\n" + "=" * 50)
    if all_passed:
        print("🎉 SETUP VERIFICATION PASSED!")
        print("Ready to begin implementation.")
    else:
        print("❌ SETUP VERIFICATION FAILED!")
        print("Please fix the issues above before continuing.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    with open("verify_setup.py", "w") as f:
        f.write(verification_script)
    
    # Make it executable
    os.chmod("verify_setup.py", 0o755)
    
    print("✅ Created setup verification script")

def create_readme():
    """Create project README"""
    
    readme_content = """# Hierarchical 2v2 Air Hockey

A sophisticated multi-agent reinforcement learning environment implementing hierarchical control for 2v2 air hockey.

## Architecture

- **High-Level Agents**: Team managers that assign tactical policies to players
- **Low-Level Agents**: Individual paddle controllers executing assigned policies  
- **Dynamic Strategy**: Real-time adaptation of team formation and roles
- **Curriculum Learning**: Progressive training from individual skills to team coordination

## Quick Start

1. **Setup Environment**:
   ```bash
   python verify_setup.py
   pip install -r requirements.txt
   ```

2. **Start Training**:
   ```bash
   python src/training/hierarchical_trainer.py --config configs/training_config.yaml
   ```

3. **Monitor Progress**:
   ```bash
   tensorboard --logdir logs/tensorboard
   ```

4. **Evaluate Model**:
   ```bash
   python src/evaluation/evaluator.py --checkpoint checkpoints/best_models/latest
   ```

## Project Structure

```
├── src/                    # Source code
│   ├── environments/       # Game environment
│   ├── agents/            # RL agents  
│   ├── training/          # Training scripts
│   └── evaluation/        # Testing & analysis
├── checkpoints/           # Saved models
├── configs/              # Configuration files
├── logs/                 # Training logs
└── results/              # Experiment results
```

## Training Stages

1. **Individual Policies** (1M steps): Train defensive, offensive, passing policies
2. **Fixed Combinations** (2M steps): Train specific policy pairs
3. **Full Hierarchical** (5M+ steps): Train complete hierarchical system

## Key Features

- Hierarchical multi-agent reinforcement learning
- Dynamic policy assignment and team coordination
- Curriculum learning with automatic progression
- Comprehensive evaluation and visualization tools
- Modular architecture for easy experimentation

## Configuration

Edit `configs/training_config.yaml` to customize:
- Training parameters
- Reward structures  
- Curriculum progression
- Agent architectures

## Development Status

- [x] Project structure created
- [ ] Base environment implementation
- [ ] Hierarchical agents
- [ ] Training pipeline
- [ ] Evaluation framework
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created project README")

def main():
    """Execute complete project setup"""
    print("🚀 Setting up Hierarchical Air Hockey Project")
    print("=" * 60)
    
    try:
        create_directory_structure()
        #copy_existing_game_files()
        create_requirements_file()
        create_config_files()
        create_initial_python_files()
        create_setup_verification()
        create_readme()
        
        print("\n" + "=" * 60)
        print("🎉 PROJECT SETUP COMPLETE!")
        print("=" * 60)
        
        print("\nNext Steps:")
        print("1. Run: pip install -r requirements.txt")
        print("2. Run: python verify_setup.py")
        print("3. Begin Phase 1.2: Environment Implementation")
        
        print("\n📁 Project Structure Created:")
        print("- Complete directory hierarchy")  
        print("- Configuration files")
        print("- Initial Python modules")
        print("- Setup verification script")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
