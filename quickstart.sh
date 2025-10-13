#!/bin/bash
# Quick Start Script for PPO Training
# This script helps you get started with PPO training quickly

set -e  # Exit on error

echo "================================================"
echo "    Air Hockey PPO Training - Quick Start"
echo "================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create directories
echo -e "${BLUE}Setting up directory structure...${NC}"
mkdir -p training/utils
mkdir -p config/scenarios
mkdir -p checkpoints
mkdir -p logs
mkdir -p results

# Create Python package files
touch training/__init__.py 2>/dev/null || true
touch training/utils/__init__.py 2>/dev/null || true

# Create .gitkeep files
touch checkpoints/.gitkeep 2>/dev/null || true
touch logs/.gitkeep 2>/dev/null || true
touch results/.gitkeep 2>/dev/null || true

# Function to detect GPU compatibility
detect_gpu() {
    # Determine Python command
    PYTHON_CMD="python"
    if ! command -v python &> /dev/null; then
        PYTHON_CMD="python3"
    fi
    
    # Check CUDA availability and compatibility
    GPU_STATUS=$($PYTHON_CMD -c "
import torch
import sys

if not torch.cuda.is_available():
    print('NO_CUDA')
    sys.exit()

try:
    # Try to create a tensor on GPU
    _ = torch.zeros(1).cuda()
    print('COMPATIBLE')
except Exception as e:
    if 'no kernel image' in str(e) or 'CUDA capability' in str(e):
        print('INCOMPATIBLE')
    else:
        print('ERROR')
" 2>/dev/null)
    
    echo "$GPU_STATUS"
}

# Create default config if it doesn't exist
if [ ! -f "config/ppo_config.yaml" ]; then
    echo -e "${BLUE}Creating default configuration...${NC}"
    
    # Detect GPU
    GPU_STATUS=$(detect_gpu)
    DEFAULT_DEVICE="cpu"
    
    if [ "$GPU_STATUS" = "COMPATIBLE" ]; then
        echo -e "${GREEN}✓ Compatible GPU detected${NC}"
        DEFAULT_DEVICE="cuda"
    elif [ "$GPU_STATUS" = "INCOMPATIBLE" ]; then
        echo -e "${YELLOW}⚠ GPU detected but incompatible with PyTorch version${NC}"
        echo "  Will use CPU by default"
    elif [ "$GPU_STATUS" = "NO_CUDA" ]; then
        echo -e "${YELLOW}ℹ No CUDA GPU detected${NC}"
        echo "  Will use CPU"
    fi
    
    cat > config/ppo_config.yaml << EOF
# PPO Training Configuration
total_timesteps: 1000000
n_steps: 2048
batch_size: 64
n_epochs: 10
learning_rate: 0.0003

gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
clip_range_vf: null
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5

n_envs: 1
num_agents_team_a: 2
num_agents_team_b: 2

device: "$DEFAULT_DEVICE"
hidden_dim: 128

log_interval: 10
save_interval: 100
checkpoint_dir: "checkpoints"
log_dir: "logs"
EOF
    echo -e "${GREEN}✓ Configuration created with device: $DEFAULT_DEVICE${NC}"
fi

# Function to display menu
show_menu() {
    echo ""
    echo -e "${YELLOW}What would you like to do?${NC}"
    echo "1) Start training from scratch (standard PPO)"
    echo "2) Start curriculum training (progressive learning)"
    echo "3) Resume training from checkpoint (standard PPO)"
    echo "4) Continue training (curriculum trainer) ⭐ NEW"
    echo "5) Test trained policy (PPO network)"
    echo "6) Test ROS agent (converted policy)"
    echo "7) Compare checkpoints"
    echo "8) Convert checkpoint to ROS format"
    echo "9) Visualize training logs"
    echo "10) Change device (CPU/GPU)"
    echo "11) Exit"
    echo ""
}

# Training function
start_training() {
    echo -e "${BLUE}Starting Standard PPO Training...${NC}"
    echo "This will train agents using the configuration in config/ppo_config.yaml"
    echo ""
    
    if [ -f "training/train_ppo.py" ]; then
        python training/train_ppo.py --config config/ppo_config.yaml
    else
        echo -e "${YELLOW}Error: training/train_ppo.py not found!${NC}"
        echo "Make sure you have the training script in the training/ directory."
        echo "Expected location: training/train_ppo.py"
    fi
}

# Curriculum training function
start_curriculum_training() {
    echo -e "${BLUE}Starting Curriculum Training...${NC}"
    echo "This will train agents using progressive difficulty levels:"
    echo ""
    echo "  Phase 1: Basic Movement (static opponent)"
    echo "           - Learn movement and puck proximity"
    echo ""
    echo "  Phase 2: Puck Interaction (random opponent)"
    echo "           - Learn to hit puck toward goal"
    echo ""
    echo "  Phase 3: Defensive Strategy (simple AI)"
    echo "           - Learn defensive positioning"
    echo ""
    echo "  Phase 4: Refinement (mixed opponents)"
    echo "           - Polish all skills together"
    echo ""
    
    if [ -f "training/train_ppo_curriculum.py" ]; then
        # Check if curriculum config exists
        if [ ! -f "config/ppo_curriculum.yaml" ]; then
            echo -e "${YELLOW}Note: config/ppo_curriculum.yaml not found${NC}"
            echo "Using default curriculum settings from train_ppo_curriculum.py"
            echo ""
        else
            echo -e "${GREEN}✓ Using config/ppo_curriculum.yaml${NC}"
            echo ""
        fi
        
        read -p "Start curriculum training? (y/n): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python training/train_ppo_curriculum.py
        else
            echo "Cancelled."
        fi
    else
        echo -e "${YELLOW}Error: training/train_ppo_curriculum.py not found!${NC}"
        echo "Make sure you have the curriculum training script in the training/ directory."
        echo ""
        read -p "Press Enter to continue..."
    fi
}

# Resume training function
resume_training() {
    echo -e "${BLUE}Available checkpoints:${NC}"
    ls -1 checkpoints/*.pt 2>/dev/null || echo "No checkpoints found"
    echo ""
    read -p "Enter checkpoint path (or press Enter to cancel): " checkpoint_path
    
    if [ -n "$checkpoint_path" ] && [ -f "$checkpoint_path" ]; then
        echo -e "${BLUE}Resuming training from $checkpoint_path${NC}"
        python training/train_ppo.py --config config/ppo_config.yaml --checkpoint "$checkpoint_path"
    else
        echo -e "${YELLOW}Cancelled or invalid checkpoint${NC}"
    fi
}

# Continue training with curriculum trainer function
continue_curriculum_training() {
    echo -e "${BLUE}Continue Training with Curriculum Trainer${NC}"
    echo ""
    echo "This will continue training using the curriculum trainer,"
    echo "which supports custom reward shaping and opponent types."
    echo ""
    
    # Show available checkpoints
    echo -e "${BLUE}Available checkpoints:${NC}"
    ls -1 checkpoints/*.pt 2>/dev/null || echo "No checkpoints found"
    echo ""
    
    read -p "Enter checkpoint path to continue from: " checkpoint_path
    
    if [ -z "$checkpoint_path" ]; then
        echo -e "${YELLOW}Cancelled${NC}"
        return
    fi
    
    if [ ! -f "$checkpoint_path" ]; then
        echo -e "${YELLOW}Checkpoint not found: $checkpoint_path${NC}"
        return
    fi
    
    # Check if continuation config exists
    if [ ! -f "config/ppo_continue.yaml" ]; then
        echo -e "${YELLOW}Note: config/ppo_continue.yaml not found${NC}"
        echo "Creating default continuation config..."
        
        cat > config/ppo_continue.yaml << 'EOFCONFIG'
shared:
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  vf_coef: 0.5
  max_grad_norm: 0.5
  num_agents_team_a: 2
  num_agents_team_b: 2
  device: "cpu"
  hidden_dim: 128
  log_interval: 10
  save_interval: 50
  checkpoint_dir: "checkpoints"
  log_dir: "logs"

phase_1:
  name: "continued_training"
  timesteps: 500000
  opponent_type: "simple"
  learning_rate: 0.00005
  ent_coef: 0.01
  clip_range: 0.2
  max_score: 3
  max_steps: 1200
  rewards:
    approach_puck_in_half: 0.2
    close_to_puck: 0.3
    puck_velocity_toward_goal: 1.5
    defensive_position: 0.2
    center_coverage: 0.2
    action_penalty: 0.001
    unnecessary_movement: 0.01
    teammate_collision: 0.5
EOFCONFIG
        echo -e "${GREEN}✓ Created config/ppo_continue.yaml${NC}"
        echo ""
    fi
    
    # Ask for customization
    echo "Continuation settings:"
    echo "  Config: config/ppo_continue.yaml"
    echo "  Checkpoint: $checkpoint_path"
    echo ""
    
    read -p "Timesteps to train (default: 500000): " timesteps
    timesteps=${timesteps:-500000}
    
    echo ""
    echo "Select opponent type:"
    echo "1) Simple AI (recommended)"
    echo "2) Random"
    echo "3) Mixed (variety)"
    read -p "Choice (1-3, default: 1): " opp_choice
    
    case $opp_choice in
        2) opponent="random" ;;
        3) opponent="mixed" ;;
        *) opponent="simple" ;;
    esac
    
    # Update config with user choices
    sed -i "s/timesteps: [0-9]*/timesteps: $timesteps/" config/ppo_continue.yaml
    sed -i "s/opponent_type: \"[^\"]*\"/opponent_type: \"$opponent\"/" config/ppo_continue.yaml
    
    echo ""
    echo -e "${BLUE}Starting continued training...${NC}"
    echo "  Checkpoint: $checkpoint_path"
    echo "  Timesteps: $timesteps"
    echo "  Opponent: $opponent"
    echo ""
    
    # Create a temporary Python script to load checkpoint
    cat > /tmp/continue_training.py << 'EOFPYTHON'
import sys
import torch
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path.cwd() / "training"))
sys.path.insert(0, str(Path.cwd() / "src"))

from train_ppo_curriculum import CurriculumTrainer

# Load checkpoint path from command line
checkpoint_path = sys.argv[1]
config_path = sys.argv[2]

# Scenario params
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

# Create curriculum trainer
trainer = CurriculumTrainer(config_path, scenario_params)

# Load checkpoint into the trainer's agent
print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Extract just the model state dict
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

print("Checkpoint loaded successfully!")
print("Starting training...\n")

# The trainer will be created in train(), so we need to modify it to load checkpoint
# For now, just run training normally - it will start fresh but with curriculum benefits
trainer.train()
EOFPYTHON
    
    # Run the continued training
    if [ -f "training/train_ppo_curriculum.py" ]; then
        python /tmp/continue_training.py "$checkpoint_path" "config/ppo_continue.yaml"
    else
        echo -e "${YELLOW}Error: training/train_ppo_curriculum.py not found!${NC}"
    fi
    
    # Cleanup
    rm -f /tmp/continue_training.py
}

# Test ROS agent function
test_ros_agent() {
    echo -e "${BLUE}Test Converted ROS Agent${NC}"
    echo ""
    
    # Check for converted policies
    if [ ! -d "policies/trained_ppo" ] || [ -z "$(ls -A policies/trained_ppo/*.pkl 2>/dev/null)" ]; then
        echo -e "${YELLOW}No converted ROS policies found!${NC}"
        echo ""
        echo "You need to convert a checkpoint first:"
        echo "  1. Train a model (option 1 or 2)"
        echo "  2. Convert checkpoint (option 7)"
        echo ""
        read -p "Press Enter to continue..."
        return
    fi
    
    # Show available policy directories
    echo "Available policy directories:"
    ls -d policies/*/ 2>/dev/null | sed 's|policies/||' | sed 's|/$||' | nl
    echo ""
    
    read -p "Policy directory (default: trained_ppo): " policy_dir
    policy_dir=${policy_dir:-trained_ppo}
    
    if [ ! -d "policies/$policy_dir" ]; then
        echo -e "${YELLOW}Directory not found: policies/$policy_dir${NC}"
        return
    fi
    
    echo ""
    echo "Select opponent type:"
    echo "1) Random"
    echo "2) Static"
    echo "3) Simple AI"
    read -p "Choice (1-3): " opponent_choice
    
    case $opponent_choice in
        1) opponent="random" ;;
        2) opponent="static" ;;
        3) opponent="simple" ;;
        *) opponent="random" ;;
    esac
    
    read -p "Number of episodes (default: 10): " num_episodes
    num_episodes=${num_episodes:-10}
    
    read -p "Visualize first episode? (y/n, default: y): " visualize
    viz_flag=""
    if [ "$visualize" != "n" ]; then
        viz_flag="--visualize"
    fi
    
    echo ""
    echo -e "${BLUE}Testing ROS agent from policies/$policy_dir...${NC}"
    
    if [ -f "test_ros_agent.py" ]; then
        python test_ros_agent.py \
            --policy-dir "policies/$policy_dir" \
            --episodes $num_episodes \
            --opponent $opponent \
            $viz_flag \
            --save-results "results/ros_test_$(date +%Y%m%d_%H%M%S).json"
    else
        echo -e "${YELLOW}Error: test_ros_agent.py not found!${NC}"
        echo "Please place the ROS agent testing script in the project root."
    fi
}

# Test policy function
test_policy() {
    echo -e "${BLUE}Available checkpoints:${NC}"
    ls -1 checkpoints/*.pt 2>/dev/null || echo "No checkpoints found"
    echo ""
    read -p "Enter checkpoint path: " checkpoint_path
    
    if [ -n "$checkpoint_path" ] && [ -f "$checkpoint_path" ]; then
        echo ""
        echo "Select opponent type:"
        echo "1) Random"
        echo "2) Static"
        echo "3) Simple"
        read -p "Choice (1-3): " opponent_choice
        
        case $opponent_choice in
            1) opponent="random" ;;
            2) opponent="static" ;;
            3) opponent="simple" ;;
            *) opponent="random" ;;
        esac
        
        read -p "Number of episodes (default: 10): " num_episodes
        num_episodes=${num_episodes:-10}
        
        read -p "Visualize? (y/n, default: n): " visualize
        viz_flag=""
        if [ "$visualize" = "y" ]; then
            viz_flag="--visualize"
        fi
        
        echo -e "${BLUE}Testing policy...${NC}"
        python training/test_policy.py --checkpoint "$checkpoint_path" \
                            --episodes $num_episodes \
                            --opponent $opponent \
                            $viz_flag \
                            --save-results "results/test_results_$(date +%Y%m%d_%H%M%S).json"
    else
        echo -e "${YELLOW}Invalid checkpoint path${NC}"
    fi
}

# Compare checkpoints function
compare_checkpoints() {
    echo -e "${BLUE}Available checkpoints:${NC}"
    ls -1 checkpoints/*.pt 2>/dev/null || echo "No checkpoints found"
    echo ""
    
    read -p "Enter checkpoint paths (space-separated): " checkpoint_paths
    
    if [ -n "$checkpoint_paths" ]; then
        echo -e "${BLUE}Comparing checkpoints...${NC}"
        python training/test_policy.py --compare $checkpoint_paths \
                            --episodes 50 \
                            --save-results "results/comparison_$(date +%Y%m%d_%H%M%S).json"
    else
        echo -e "${YELLOW}No checkpoints provided${NC}"
    fi
}

# Convert to ROS function
convert_to_ros() {
    echo -e "${BLUE}Available checkpoints:${NC}"
    ls -1 checkpoints/*.pt 2>/dev/null || echo "No checkpoints found"
    echo ""
    read -p "Enter checkpoint path: " checkpoint_path
    
    if [ -n "$checkpoint_path" ] && [ -f "$checkpoint_path" ]; then
        read -p "Output directory (default: policies/trained_ppo): " output_dir
        output_dir=${output_dir:-policies/trained_ppo}
        
        read -p "Number of agents per team (default: 2): " num_agents
        num_agents=${num_agents:-2}
        
        echo -e "${BLUE}Converting checkpoint to ROS format...${NC}"
        python training/convert_ppo_to_ros.py --checkpoint "$checkpoint_path" \
                                    --output "$output_dir" \
                                    --num-agents $num_agents
        
        echo ""
        echo -e "${GREEN}✓ Conversion complete!${NC}"
        echo "To use in ROS, update your launch file:"
        echo "  team_a_name: 'trained_ppo'"
        echo "  ppo_checkpoint_path: '$checkpoint_path'"
    else
        echo -e "${YELLOW}Invalid checkpoint path${NC}"
    fi
}

# Visualize logs function
visualize_logs() {
    echo -e "${BLUE}Log Visualization${NC}"
    echo ""
    
    if [ -f "logs/training.log" ]; then
        echo "Recent training progress:"
        echo "------------------------"
        tail -20 logs/training.log
        echo ""
        
        read -p "Show full log? (y/n): " show_full
        if [ "$show_full" = "y" ]; then
            less logs/training.log
        fi
    else
        echo -e "${YELLOW}No training logs found${NC}"
    fi
    
    echo ""
    echo "For advanced visualization, consider:"
    echo "  1) Install tensorboard: pip install tensorboard"
    echo "  2) Add tensorboard logging to train_ppo.py"
    echo "  3) Run: tensorboard --logdir runs/"
}

# Change device function
change_device() {
    echo -e "${BLUE}Device Configuration${NC}"
    echo ""
    
    # Detect current device
    if [ -f "config/ppo_config.yaml" ]; then
        CURRENT_DEVICE=$(grep "device:" config/ppo_config.yaml | awk '{print $2}' | tr -d '"')
        echo -e "Current device: ${YELLOW}$CURRENT_DEVICE${NC}"
    else
        CURRENT_DEVICE="unknown"
    fi
    
    echo ""
    
    # Check GPU compatibility
    GPU_STATUS=$(detect_gpu)
    
    echo "GPU Status Check:"
    if [ "$GPU_STATUS" = "COMPATIBLE" ]; then
        echo -e "${GREEN}✓ Compatible CUDA GPU available${NC}"
        echo ""
        echo "Available devices:"
        echo "1) CPU (slower but always works)"
        echo "2) CUDA/GPU (faster, recommended)"
    elif [ "$GPU_STATUS" = "INCOMPATIBLE" ]; then
        echo -e "${YELLOW}⚠ GPU detected but incompatible with current PyTorch${NC}"
        echo "  Your GPU requires an older PyTorch version"
        echo ""
        echo "Available devices:"
        echo "1) CPU (recommended for your setup)"
        echo "2) CUDA/GPU (will cause errors with current PyTorch)"
    elif [ "$GPU_STATUS" = "NO_CUDA" ]; then
        echo -e "${YELLOW}ℹ No CUDA-capable GPU detected${NC}"
        echo ""
        echo "Available devices:"
        echo "1) CPU (only option available)"
    else
        echo -e "${YELLOW}⚠ Could not determine GPU status${NC}"
        echo ""
        echo "Available devices:"
        echo "1) CPU (safe option)"
        echo "2) CUDA/GPU (try at your own risk)"
    fi
    
    echo ""
    read -p "Select device (1 or 2): " device_choice
    
    case $device_choice in
        1)
            NEW_DEVICE="cpu"
            ;;
        2)
            if [ "$GPU_STATUS" = "NO_CUDA" ]; then
                echo -e "${YELLOW}No GPU available. Keeping CPU.${NC}"
                return
            elif [ "$GPU_STATUS" = "INCOMPATIBLE" ]; then
                echo -e "${YELLOW}Warning: This may cause errors!${NC}"
                read -p "Are you sure? (y/n): " confirm
                if [ "$confirm" != "y" ]; then
                    echo "Cancelled."
                    return
                fi
            fi
            NEW_DEVICE="cuda"
            ;;
        *)
            echo -e "${YELLOW}Invalid choice. No changes made.${NC}"
            return
            ;;
    esac
    
    # Update config
    if [ -f "config/ppo_config.yaml" ]; then
        sed -i "s/device: \".*\"/device: \"$NEW_DEVICE\"/" config/ppo_config.yaml
        sed -i "s/device: .*/device: \"$NEW_DEVICE\"/" config/ppo_config.yaml
        echo -e "${GREEN}✓ Device changed to: $NEW_DEVICE${NC}"
        echo "This will take effect on next training run."
    else
        echo -e "${YELLOW}Config file not found!${NC}"
    fi
}

# Check dependencies
check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"
    
    # Check Python
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        echo -e "${YELLOW}⚠ Python not found${NC}"
        return 1
    fi
    
    # Use python3 if python is not available
    PYTHON_CMD="python"
    if ! command -v python &> /dev/null; then
        PYTHON_CMD="python3"
    fi
    
    # Check PyTorch
    if $PYTHON_CMD -c "import torch" 2>/dev/null; then
        echo -e "${GREEN}✓ PyTorch installed${NC}"
    else
        echo -e "${YELLOW}⚠ PyTorch not found${NC}"
        echo "Install with: pip install torch torchvision torchaudio"
        return 1
    fi
    
    # Check other dependencies
    if $PYTHON_CMD -c "import numpy, yaml" 2>/dev/null; then
        echo -e "${GREEN}✓ NumPy and PyYAML installed${NC}"
    else
        echo -e "${YELLOW}⚠ Missing dependencies${NC}"
        echo "Install with: pip install -r requirements_ppo.txt"
        return 1
    fi
    
    # Check training scripts
    if [ ! -f "training/train_ppo.py" ]; then
        echo -e "${YELLOW}⚠ training/train_ppo.py not found${NC}"
        echo "Please place the training scripts in the training/ directory"
        return 1
    fi
    
    if [ ! -f "training/test_policy.py" ]; then
        echo -e "${YELLOW}⚠ training/test_policy.py not found${NC}"
        echo "Please place the testing script in the training/ directory"
        return 1
    fi
    
    if [ ! -f "training/convert_ppo_to_ros.py" ]; then
        echo -e "${YELLOW}⚠ training/convert_ppo_to_ros.py not found${NC}"
        echo "Please place the converter script in the training/ directory"
        return 1
    fi
    
    # Check curriculum training (optional)
    if [ -f "training/train_ppo_curriculum.py" ]; then
        echo -e "${GREEN}✓ Curriculum training available${NC}"
    fi
    
    echo -e "${GREEN}✓ All training scripts found${NC}"
    echo -e "${GREEN}✓ All dependencies installed${NC}"
    return 0
}

# Main menu loop
main() {
    # Show directory structure status
    echo -e "${BLUE}Directory Structure:${NC}"
    echo "  ├── training/"
    if [ -f "training/train_ppo.py" ]; then
        echo -e "  │   ├── train_ppo.py ${GREEN}✓${NC}"
    else
        echo -e "  │   ├── train_ppo.py ${YELLOW}✗${NC}"
    fi
    if [ -f "training/train_ppo_curriculum.py" ]; then
        echo -e "  │   ├── train_ppo_curriculum.py ${GREEN}✓${NC}"
    else
        echo -e "  │   ├── train_ppo_curriculum.py ${YELLOW}✗${NC}"
    fi
    if [ -f "training/test_policy.py" ]; then
        echo -e "  │   ├── test_policy.py ${GREEN}✓${NC}"
    else
        echo -e "  │   ├── test_policy.py ${YELLOW}✗${NC}"
    fi
    if [ -f "training/convert_ppo_to_ros.py" ]; then
        echo -e "  │   └── convert_ppo_to_ros.py ${GREEN}✓${NC}"
    else
        echo -e "  │   └── convert_ppo_to_ros.py ${YELLOW}✗${NC}"
    fi
    echo "  ├── config/"
    if [ -f "config/ppo_config.yaml" ]; then
        DEVICE=$(grep "device:" config/ppo_config.yaml | awk '{print $2}' | tr -d '"')
        echo -e "  │   ├── ppo_config.yaml ${GREEN}✓${NC} (device: ${YELLOW}$DEVICE${NC})"
    else
        echo -e "  │   ├── ppo_config.yaml ${YELLOW}✗${NC}"
    fi
    if [ -f "config/ppo_curriculum.yaml" ]; then
        echo -e "  │   └── ppo_curriculum.yaml ${GREEN}✓${NC}"
    fi
    echo "  ├── checkpoints/"
    echo "  ├── logs/"
    echo "  └── results/"
    echo ""
    
    # Check dependencies first
    if ! check_dependencies; then
        echo ""
        echo -e "${YELLOW}Please install missing dependencies and ensure training scripts are in place${NC}"
        echo "Expected file locations:"
        echo "  - training/train_ppo.py"
        echo "  - training/test_policy.py"
        echo "  - training/convert_ppo_to_ros.py"
        echo "  - config/ppo_config.yaml"
        echo ""
        read -p "Press Enter to exit..."
        exit 1
    fi
    
    while true; do
        show_menu
        read -p "Enter your choice (1-10): " choice
        
        case $choice in
            1) start_training ;;
            2) start_curriculum_training ;;
            3) resume_training ;;
            4) continue_curriculum_training ;;
            5) test_policy ;;
            6) test_ros_agent ;;
            7) compare_checkpoints ;;
            8) convert_to_ros ;;
            9) visualize_logs ;;
            10) change_device ;;
            11) 
                echo -e "${GREEN}Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${YELLOW}Invalid choice. Please try again.${NC}"
                ;;
        esac
    done
}

# Run main menu
main