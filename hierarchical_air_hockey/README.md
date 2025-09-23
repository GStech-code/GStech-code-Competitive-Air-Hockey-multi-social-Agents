# Hierarchical 2v2 Air Hockey

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
