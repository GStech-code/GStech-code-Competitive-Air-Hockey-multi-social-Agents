# First run
To run air hockey ros, follow the instructions below.
First, have ROS 2 installed on ubuntu operating system
Note: this project was tested on ROS 2 jazzy distribution on ubuntu 24.04 LTS (via WSL) operating system.
You may be required to have certain ROS 2 related installations besides what is specified in here.

## Run the following commands after cloning:

### if you haven't already defined python environment:
```
python3 -m venv .venv 

source .venv/bin/activate
```
### Install requirements:
```
pip install -r requirements.txt
```

### Build ROS install:
```
source /opt/ros/<ROS2_DISTRO>/setup.bash
chmod +x src/air_hockey_ros/air_hockey_ros/agent_node.py
chmod +x src/air_hockey_ros/air_hockey_ros/game_manager_node.py
colcon build --symlink-install --packages-select air_hockey_ros
source install/setup.bash
```
###### Replace <ROS2_DISTRO> with your installed ROS 2 version, e.g. humble (Ubuntu 22.04) or jazzy (Ubuntu 24.04).
#### If you add files, use the colcon build & source functions again.



#### For threaded / multi processing files, need to provide permissions:
```
chmod +x path/to/script.py
```
#### To provide editing permissions when files are managed in git, this command is supposed to help:
```
git update-index --chmod=+x path/to/script.py
```
### To delete install package:
```
rm -rf build/ install/ log/
```
## Run:
### Running the simulation
#### Single game
```
ros2 launch air_hockey_ros single_game.launch.py
```
#### Multiple games
```
ros2 launch air_hockey_ros multi_game.launch.py   games_config:=games_configs/1.yaml
```
For multiple games you may choose a different config yaml and edit or alter existing ones

currently end run by SIGINT (ctrl + c)

### Checking if nodes are alive:
```
ps -ef | grep game_manager_node.py

ps -ef | grep agent_node.py
```
#### You should see only one line returned per each command after finishing the game.
#### If nodes are still alive, end them with the kill command.

### Running replay
To run the replay, you may need to source the environment.
You will need to provide the path to the log file (.log), the path to the scenario file (.yaml) and the hz (usually 60)
Example command:

```
python replay.py --log game_logs/tournament_games/default_scenario--long_short_2-vs-two_capped_neural_2--001.log --scenario src/air_hockey_ros/game_scenarios/default_scenario.yaml```
```
### Analysis
We have simulated 71 games. Logs exist under  ```game_logs\tournament_games```

You can see the analysis at ```game_logs\tournament_analysis.ipynb```

Check out 5 examples from the tournament:
https://www.youtube.com/watch?v=uOAevuGi2G4

## 🤖 Training Neural Network Agents

This project includes PPO (Proximal Policy Optimization) training for neural network agents.

### Quick Start
```bash
# Install training dependencies
pip install -r requirements_ppo.txt

# Run interactive training menu
chmod +x quickstart.sh
./quickstart.sh
```

### Manual Training
```bash
# Train agents
python training/train_ppo.py --config config/ppo_config.yaml

# Test trained policy
python training/test_policy.py --checkpoint checkpoints/ppo_checkpoint_1000.pt --visualize

# Convert for ROS deployment
python training/convert_ppo_to_ros.py --checkpoint checkpoints/ppo_checkpoint_1000.pt --output policies/trained
```

For detailed training documentation, see [TRAINING.md](TRAINING.md)


cd /mnt/c/Users/galsa/source/repos/AirHockey/GStech-code-Competitive-Air-Hockey-multi-social-Agents

source .venv/bin/activate

./quickstart.sh
