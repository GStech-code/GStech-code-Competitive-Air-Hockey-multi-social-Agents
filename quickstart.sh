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
    echo "1) Start training from scratch"
    echo "2) Resume training from checkpoint"
    echo "3) Test trained policy (PPO network)"
    echo "4) Test ROS agent (converted policy)"
    echo "5) Compare checkpoints"
    echo "6) Convert checkpoint to ROS format"
    echo "7) Visualize training logs"
    echo "8) Change device (CPU/GPU)"
    echo "9) Exit"
    echo ""
}

# Training function
start_training() {
    echo -e "${BLUE}Starting PPO training...${NC}"
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

# Test ROS agent function
test_ros_agent() {
    echo -e "${BLUE}Test Converted ROS Agent${NC}"
    echo ""
    
    # Check for converted policies
    if [ ! -d "policies/trained_ppo" ] || [ -z "$(ls -A policies/trained_ppo/*.pkl 2>/dev/null)" ]; then
        echo -e "${YELLOW}No converted ROS policies found!${NC}"
        echo ""
        echo "You need to convert a checkpoint first:"
        echo "  1. Train a model (option 1)"
        echo "  2. Convert checkpoint (option 6)"
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
        echo -e "  │   └── ppo_config.yaml ${GREEN}✓${NC} (device: ${YELLOW}$DEVICE${NC})"
    else
        echo -e "  │   └── ppo_config.yaml ${YELLOW}✗${NC}"
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
        read -p "Enter your choice (1-9): " choice
        
        case $choice in
            1) start_training ;;
            2) resume_training ;;
            3) test_policy ;;
            4) test_ros_agent ;;
            5) compare_checkpoints ;;
            6) convert_to_ros ;;
            7) visualize_logs ;;
            8) change_device ;;
            9) 
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