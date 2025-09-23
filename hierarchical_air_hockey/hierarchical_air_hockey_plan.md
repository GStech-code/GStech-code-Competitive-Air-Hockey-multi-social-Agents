# Hierarchical 2v2 Air Hockey: Complete Development Plan

## Project Overview

This project implements a hierarchical multi-agent reinforcement learning system for 2v2 air hockey, where:

- **High-level agents**: Team managers that assign policies (defensive, offensive, passing) to individual players
- **Low-level agents**: Individual paddle controllers that execute assigned policies
- **Dynamic strategy**: Teams can adapt their formation and strategy in real-time
- **Coordination**: Players learn to coordinate through both explicit policy assignment and implicit communication

### Architecture
```
Team Blue Agent (High-Level)    Team Red Agent (High-Level)
     |                               |
     ├─ Blue_A Policy Assignment     ├─ Red_A Policy Assignment
     └─ Blue_B Policy Assignment     └─ Red_B Policy Assignment
          |                               |
    Blue_A Paddle Agent           Red_A Paddle Agent
    Blue_B Paddle Agent           Red_B Paddle Agent
    (Low-Level Controllers)       (Low-Level Controllers)
```

## Phase 1: Project Setup and Environment Enhancement

### Step 1.1: Create Project Structure

```bash
hierarchical_air_hockey/
├── src/
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── base_env.py
│   │   ├── hierarchical_env.py
│   │   └── game_core/
│   │       ├── game2v2.py
│   │       ├── disc.py
│   │       ├── paddle.py
│   │       └── helper_functions.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── high_level_agent.py
│   │   ├── low_level_agents.py
│   │   └── hierarchical_policies.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── hierarchical_trainer.py
│   │   ├── curriculum_learning.py
│   │   └── reward_systems.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── visualization.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logging_utils.py
├── checkpoints/
│   ├── high_level/
│   │   ├── blue_team/
│   │   └── red_team/
│   └── low_level/
│       ├── defensive/
│       ├── offensive/
│       └── passing/
├── configs/
│   ├── training_config.yaml
│   ├── environment_config.yaml
│   └── agent_config.yaml
├── logs/
├── results/
├── tests/
└── requirements.txt
```

### Step 1.2: Enhanced Environment Implementation

Create `src/environments/hierarchical_env.py`:

Key features:
- Separate observation spaces for high-level and low-level agents
- Policy assignment interface
- Team coordination rewards
- Dynamic role switching

### Step 1.3: Policy Definitions

Define three core policies:
1. **Defensive Policy**: Focus on goal protection and interception
2. **Offensive Policy**: Aggressive disc pursuit and shooting
3. **Passing Policy**: Team coordination and disc distribution

## Phase 2: Hierarchical Agent Architecture

### Step 2.1: High-Level Team Agents

High-level agents observe:
- Overall game state
- Team formation
- Disc position and velocity
- Opponent positions
- Current policy assignments

Actions:
- Assign policies to each team member
- Strategic formation commands
- Priority targets

### Step 2.2: Low-Level Paddle Agents

Each paddle agent receives:
- Local observations (disc, nearby players, goal)
- Assigned policy from high-level agent
- Team communication signals

Actions:
- Movement commands (x, y velocity)
- Execution intensity (aggressive/conservative)

### Step 2.3: Communication Protocol

Implement communication channels:
- High-level to low-level: Policy assignments, priority targets
- Low-level to high-level: Status reports, capability feedback
- Peer-to-peer: Implicit coordination through shared observations

## Phase 3: Training Infrastructure

### Step 3.1: Curriculum Learning Setup

Training progression:
1. **Stage 1**: Individual policy training (1000 episodes each)
2. **Stage 2**: Fixed policy combinations (2000 episodes)
3. **Stage 3**: Dynamic policy switching (3000 episodes)
4. **Stage 4**: Full hierarchical training (5000+ episodes)

### Step 3.2: Reward System Design

**High-Level Rewards**:
- Team coordination bonus
- Strategic positioning rewards
- Adaptive strategy rewards
- Long-term game state improvements

**Low-Level Rewards**:
- Policy-specific objectives
- Individual skill rewards
- Team support bonuses
- Execution quality metrics

### Step 3.3: Multi-Agent Training Setup

Use Ray RLlib with:
- Separate policy networks for high and low levels
- Shared experience replay between similar policies
- Asynchronous training for different hierarchy levels

## Phase 4: Implementation Details

### Step 4.1: Environment Code Structure

```python
class HierarchicalAirHockeyEnv(MultiAgentEnv):
    def __init__(self, config):
        # Initialize base game environment
        # Setup hierarchical observation/action spaces
        # Configure communication channels
        
    def step(self, actions):
        # Process high-level decisions first
        # Update policy assignments
        # Execute low-level actions
        # Calculate hierarchical rewards
        
    def get_observations(self):
        # High-level: strategic game state
        # Low-level: local tactical information
        # Communication: inter-agent messages
```

### Step 4.2: Training Configuration

```yaml
# configs/training_config.yaml
training:
  algorithm: "MADDPG"  # or PPO with hierarchical modifications
  total_timesteps: 10_000_000
  
hierarchical:
  levels: 2
  high_level_frequency: 10  # steps between high-level decisions
  low_level_frequency: 1   # continuous low-level control
  
curriculum:
  enable: true
  stages:
    - name: "individual_policies"
      duration: 1_000_000
      focus: "low_level_only"
    - name: "fixed_combinations" 
      duration: 2_000_000
      focus: "policy_combinations"
    - name: "adaptive_strategies"
      duration: 5_000_000
      focus: "full_hierarchy"

rewards:
  high_level:
    team_coordination: 10.0
    strategic_positioning: 5.0
    adaptive_bonus: 15.0
  low_level:
    policy_adherence: 3.0
    individual_skill: 5.0
    team_support: 7.0
```

## Phase 5: Training Process

### Step 5.1: Environment Preparation

1. **Install dependencies**:
```bash
pip install ray[rllib] torch gymnasium pygame pettingzoo tensorboard
```

2. **Initialize training environment**:
```bash
cd hierarchical_air_hockey
python src/training/setup_training.py
```

### Step 5.2: Curriculum Training Stages

**Stage 1: Individual Policy Training**
```bash
python src/training/train_policies.py --stage individual --policies defensive,offensive,passing
```

Train each policy independently:
- Defensive agents learn goal protection
- Offensive agents learn scoring
- Passing agents learn coordination

**Stage 2: Policy Combination Training**
```bash
python src/training/train_combinations.py --stage combinations
```

Train fixed policy combinations:
- Both defensive (defensive formation)
- Both offensive (aggressive formation) 
- One defensive, one offensive (balanced formation)
- Passing combinations (coordination formation)

**Stage 3: Hierarchical Training**
```bash
python src/training/train_hierarchical.py --stage full
```

Full hierarchical training with:
- High-level agents learning strategy selection
- Low-level agents adapting to dynamic assignments
- Communication protocol optimization

### Step 5.3: Training Monitoring

Real-time monitoring tools:
```bash
# Start TensorBoard
tensorboard --logdir logs/

# Training progress dashboard
python src/evaluation/training_dashboard.py

# Live game visualization
python src/evaluation/live_viewer.py --checkpoint latest
```

Key metrics to track:
- Individual policy performance
- Team coordination scores
- Strategy adaptation success rate
- Win rate progression
- Communication effectiveness

## Phase 6: Testing and Evaluation

### Step 6.1: Policy Evaluation Framework

Create comprehensive testing suite:

```python
# src/evaluation/evaluator.py
class HierarchicalEvaluator:
    def evaluate_individual_policies(self, checkpoint_path):
        # Test each policy in isolation
        
    def evaluate_team_coordination(self, checkpoint_path):
        # Test team formation and cooperation
        
    def evaluate_strategy_adaptation(self, checkpoint_path):
        # Test dynamic strategy switching
        
    def evaluate_vs_baselines(self, checkpoint_path):
        # Test against rule-based opponents
```

### Step 6.2: Visual Testing Interface

```python
# src/evaluation/visual_tester.py
class VisualTester:
    def __init__(self, checkpoint_path):
        self.env = HierarchicalAirHockeyEnv({"render_mode": "human"})
        self.agents = self.load_agents(checkpoint_path)
        
    def run_interactive_test(self):
        # Real-time policy visualization
        # Strategy assignment display
        # Performance metrics overlay
        
    def policy_comparison_mode(self):
        # Side-by-side policy comparison
        # A/B testing interface
```

### Step 6.3: Testing Commands

```bash
# Test trained models
python src/evaluation/test_hierarchical.py --checkpoint checkpoints/hierarchical/final_model

# Interactive testing with visualization
python src/evaluation/visual_tester.py --checkpoint checkpoints/hierarchical/final_model --interactive

# Benchmark against baselines
python src/evaluation/benchmark.py --model checkpoints/hierarchical/final_model --opponents random,rule_based

# Policy analysis
python src/evaluation/analyze_policies.py --checkpoint checkpoints/hierarchical/final_model
```

## Phase 7: Advanced Features

### Step 7.1: Opponent Modeling

Implement opponent adaptation:
- Learn opponent strategies
- Counter-strategy development
- Adaptive formation responses

### Step 7.2: Transfer Learning

Enable knowledge transfer:
- Pre-trained policy initialization
- Cross-team learning
- Skill composition

### Step 7.3: Human-AI Interaction

Create human playable modes:
- Human + AI teammate vs AI team
- Human team manager mode
- Collaborative training

## Phase 8: Deployment and Analysis

### Step 8.1: Model Optimization

Optimize for deployment:
- Model compression
- Inference speed optimization
- Memory usage reduction

### Step 8.2: Comprehensive Analysis

Generate detailed analysis:
- Strategy emergence patterns
- Communication protocol effectiveness
- Coordination mechanism analysis
- Performance scaling with team size

### Step 8.3: Documentation and Reproducibility

Complete documentation:
- Training recipes
- Hyperparameter sensitivity analysis
- Reproducibility guidelines
- Performance benchmarks

## Complete Training Pipeline

### Quick Start Commands

```bash
# 1. Setup project
git clone <repository>
cd hierarchical_air_hockey
pip install -r requirements.txt

# 2. Initialize training
python setup.py --create-structure
python setup.py --verify-environment

# 3. Start curriculum training
python train.py --curriculum --visualize --save-checkpoints

# 4. Monitor training
tensorboard --logdir logs/
python monitor_training.py --live

# 5. Test trained model
python test.py --checkpoint checkpoints/hierarchical/best_model --interactive --record

# 6. Analyze results
python analyze_results.py --checkpoint checkpoints/hierarchical/best_model
```

### Training Time Estimates

- **Individual Policies**: 2-4 hours per policy (3 policies)
- **Policy Combinations**: 6-8 hours
- **Full Hierarchical**: 12-24 hours
- **Total Training Time**: 24-40 hours on modern GPU

### Expected Performance Metrics

Success indicators:
- **Individual Policy Success Rate**: >80% task completion
- **Team Coordination Score**: >0.85 formation maintenance
- **Strategy Adaptation Speed**: <50 steps to optimal strategy
- **Overall Win Rate**: >70% vs baseline opponents

## Key Implementation Notes

1. **Communication Architecture**: Implement explicit communication channels between hierarchical levels
2. **Curriculum Design**: Gradual complexity increase prevents training instability
3. **Reward Shaping**: Careful balance between individual and team rewards
4. **Observation Design**: High-level agents need strategic information, low-level agents need tactical details
5. **Model Architecture**: Separate networks for different hierarchy levels and policies
6. **Training Stability**: Use experience replay and target networks for stable learning

This hierarchical approach enables sophisticated team strategies while maintaining individual skill development, creating a more realistic and engaging air hockey AI system.