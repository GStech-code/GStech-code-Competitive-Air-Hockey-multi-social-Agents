#!/usr/bin/env python3
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json, yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# ---------- helpers ----------
def _load_yaml(p: str) -> dict:
    p = Path(__file__).resolve().parent / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _flatten_rules(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flatten_rules(v))
        else:
            out[k] = v
    return out

def _resolve_scenario_path(base: Path, scenario_path: str) -> str:
    p = Path(scenario_path)
    if p.is_file():
        return str(p)
    # relative to CWD
    cwd = Path.cwd() / scenario_path
    if cwd.is_file():
        return str(cwd)
    # relative to this launch file
    here_rel = base.parent / scenario_path
    if here_rel.is_file():
        return str(here_rel)
    # package share fallback
    try:
        share = Path(get_package_share_directory("air_hockey_ros"))
        guess = share / "game_scenarios" / Path(scenario_path).name
        if guess.is_file():
            return str(guess)
    except Exception:
        pass
    return scenario_path  # let ROS print a clear error

def _expand_games(cfg: dict, base_file: Path) -> Tuple[dict, List[dict]]:
    static = cfg.get("static", {}) or {}
    games = cfg.get("games", []) or []
    expanded = []
    for g in games:
        rep = int(g.get("repetition", 1))
        for _ in range(rep):
            item = dict(g)
            item["scenario_file"] = _resolve_scenario_path(base_file, g["scenario_file"])
            expanded.append(item)
    return static, expanded

def _node_args_from_static(static: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Pull 'sim_name' (required) and convert all OTHER static keys
    into 'key::=value' strings (bools lowercased, complex -> JSON).
    """
    if "sim_name" not in static:
        raise ValueError("static.sim_name is required in games_config")

    sim_name = str(static["sim_name"])
    args = []
    for k, v in static.items():
        if k == "sim_name" or k == "training_cmd":
            continue
        if isinstance(v, bool):
            args.append(f"{k}::={'true' if v else 'false'}")
        elif isinstance(v, (int, float)) or v is None or isinstance(v, str):
            vv = "null" if v is None else str(v)
            args.append(f"{k}::={vv}")
        else:
            args.append(f"{k}::='{json.dumps(v, separators=(',', ':'))}'")
    return sim_name, args

def _make_start_req(scenario_file: str,
                    team_a: str, team_b: str,
                    num_a: int, num_b: int) -> str:
    with open(scenario_file, "r", encoding="utf-8") as f:
        scn = yaml.safe_load(f) or {}
    scenario_name = scn.pop("name", Path(scenario_file).stem)
    rules_json = json.dumps(_flatten_rules(scn), separators=(",", ":"))
    return (
        "{"
        f"scenario_name: '{scenario_name}', "
        f"team_a_name: '{team_a}', "
        f"team_b_name: '{team_b}', "
        f"num_agents_team_a: {num_a}, "
        f"num_agents_team_b: {num_b}, "
        f"rules: '{rules_json}'"
        "}"
    )

# ---------- sequence builder ----------
def _build_sequence(context, *args, **kwargs):
    cfg_path = LaunchConfiguration("games_config").perform(context)
    start_service = LaunchConfiguration("start_service").perform(context)
    game_result_topic = LaunchConfiguration("game_result_topic").perform(context)  # not used for gating; kept for logs
    training_cmd = LaunchConfiguration("training_cmd").perform(context)  # default "", can be overridden

    cfg = _load_yaml(cfg_path)
    base_file = Path(__file__).resolve()
    static, games = _expand_games(cfg, base_file)
    sim_name, static_node_args = _node_args_from_static(static)

    # Prefer config's training_cmd if present
    if "training_cmd" in static and isinstance(static["training_cmd"], str):
        training_cmd = static["training_cmd"]

    actions = []

    prev_game_node = None  # for chaining with OnProcessExit

    for i, g in enumerate(games):
        # Build the node for this game
        gm_node = Node(
            package="air_hockey_ros",
            executable="game_manager_node.py",
            name=f"game_manager_{i+1}",
            output="screen",
            # oneshot=True is default inside your node; we do NOT pass it.
            arguments=[sim_name, *static_node_args],
        )

        # Prepare the StartGame request for this game
        start_req = _make_start_req(
            scenario_file=g["scenario_file"],
            team_a=str(g["team_a_name"]),
            team_b=str(g["team_b_name"]),
            num_a=int(g["num_agents_team_a"]),
            num_b=int(g["num_agents_team_b"]),
        )

        # Timer to call StartGame a moment after the node is up
        start_timer = TimerAction(
            period=2.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "ros2", "service", "call",
                        start_service,                # e.g. "/start_game"
                        "air_hockey_ros/srv/StartGame",
                        start_req,
                    ],
                    output="screen",
                )
            ],
        )

        if i == 0:
            # First game starts immediately
            actions.append(gm_node)
            actions.append(start_timer)
        else:
            # Chain: after previous game process exits -> training (optional) -> start this game
            chain_actions = []

            if training_cmd:  # optional training between games
                chain_actions.append(
                    ExecuteProcess(
                        cmd=["bash", "-lc", training_cmd],
                        shell=False,
                        output="screen",
                    )
                )

            chain_actions.extend([gm_node, start_timer])

            actions.append(
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=prev_game_node,
                        on_exit=chain_actions,
                    )
                )
            )

        prev_game_node = gm_node

    # After the last game exits -> shutdown the launch (clean exit, no SIGINT needed)
    if prev_game_node is not None:
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=prev_game_node,
                    on_exit=[Shutdown(reason="All games complete")],
                )
            )
        )

    return actions

# ---------- launch description ----------
def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "games_config",
            description="Path to YAML with 'static' and 'games' sections."
        ),
        DeclareLaunchArgument(
            "start_service",
            default_value=TextSubstitution(text="/start_game"),
            description="StartGame service name."
        ),
        DeclareLaunchArgument(
            "game_result_topic",
            default_value=TextSubstitution(text="/game_result"),
            description="(Optional) GameResult topic for logs/compat."
        ),
        DeclareLaunchArgument(
            "training_cmd",
            default_value=TextSubstitution(text=""),
            description="Optional shell command to run between games."
        ),
        OpaqueFunction(function=_build_sequence),
    ])
