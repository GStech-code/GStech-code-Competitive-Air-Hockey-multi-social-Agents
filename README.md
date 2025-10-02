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
source /opt/ros/jazzy/setup.bash
chmod +x src/air_hockey_ros/air_hockey_ros/agent_node.py
chmod +x src/air_hockey_ros/air_hockey_ros/game_manager_node.py
colcon build --symlink-install --packages-select air_hockey_ros
source install/setup.bash
```
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
Example command:

```
python replay.py --log game_logs/game_log_1.log --scenario src/air_hockey_ros/game_scenarios/simple_scenario.yaml --hz 60
```