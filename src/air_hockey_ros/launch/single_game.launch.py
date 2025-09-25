from ament_index_python.packages import get_package_share_directory
import os
from pathlib import Path
import json, yaml

share = get_package_share_directory("air_hockey_ros")
scenario_path = os.path.join(share, "game_scenarios", "simple_scenario.yaml")

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression, TextSubstitution
from launch_ros.actions import Node

# --- helpers (run inside OpaqueFunction) ---
def _flatten_rules(scn: dict) -> dict:
    rules = {}
    for key, value in scn.items():
        if isinstance(value, dict):
            rules.update(_flatten_rules(value))
        else:
            rules[key] = value
    return rules

def _resolve_scenario_path(context, scenario_file_lc):
    val = scenario_file_lc.perform(context)
    here = Path(__file__).resolve()
    candidates = [
        Path(val),
        Path.cwd() / val,
        here.parent.parent / "game_scenarios" / "simple_scenario.yaml",
    ]
    if len(here.parents) >= 4:
        candidates.append(here.parents[3] / "src" / "game_scenarios" / "simple_scenario.yaml")
    for c in candidates:
        if c.is_file():
            return str(c)
    return val

def _build_actions(context, *args, **kwargs):
    scenario_file_lc = LaunchConfiguration("scenario_file")
    sim_cmd_lc = LaunchConfiguration("sim_cmd")

    scenario_path = _resolve_scenario_path(context, scenario_file_lc)
    with open(scenario_path, "r", encoding="utf-8") as f:
        scn = yaml.safe_load(f)

    scenario_name = scn.pop("name", "simple_scenario")
    rules_dict = _flatten_rules(scn)
    rules_json = json.dumps(rules_dict, separators=(",", ":"))

    team_a = "free_simple"
    team_b = "simple"
    num_a = 2
    num_b = 2

    # YAML request; quote strings explicitly
    req = (
        "{"
        f"scenario_name: '{scenario_name}', "
        f"team_a_name: '{team_a}', "
        f"team_b_name: '{team_b}', "
        f"num_agents_team_a: {num_a}, "
        f"num_agents_team_b: {num_b}, "
        f"rules: '{rules_json}'"
        "}"
    )

    actions = []

    actions.append(
        ExecuteProcess(
            cmd=["bash", "-lc", sim_cmd_lc.perform(context)],
            shell=False,
            output="screen",
            condition=IfCondition(PythonExpression(["'", sim_cmd_lc, "' != ''"])),
        )
    )

    # Pass constructor arg as CLI arg: simulation_name='mock'
    actions.append(
        Node(
            package="air_hockey_ros",
            executable="game_manager_node.py",
            name="game_manager",
            output="screen",
            arguments=["base", "view::=true", "log_team_a::=false", "log_team_b::=false"],  # alternative: ["mock", "use_physics::=true"]
        )
    )

    actions.append(
        TimerAction(
            period=2.5,  # small delay to ensure service is available
            actions=[
                ExecuteProcess(
                    cmd=[
                        "ros2", "service", "call",
                        "/start_game",
                        "air_hockey_ros/srv/StartGame",
                        req,
                    ],
                    output="screen",
                )
            ],
        )
    )

    return actions

def generate_launch_description():
    sim_cmd       = LaunchConfiguration("sim_cmd")
    scenario_file = LaunchConfiguration("scenario_file")

    return LaunchDescription([
        DeclareLaunchArgument("sim_cmd", default_value=TextSubstitution(text="")),
        DeclareLaunchArgument("scenario_file", default_value=TextSubstitution(text="game_scenarios/simple_scenario.yaml")),
        OpaqueFunction(function=_build_actions),
    ])
