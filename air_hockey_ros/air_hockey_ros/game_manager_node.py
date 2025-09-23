#!/usr/bin/env python3
from typing import Dict, List, Literal, Tuple
import subprocess
import pickle
import tempfile, shutil
import os
import sys
import threading
import signal
import time
import logging
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from builtin_interfaces.msg import Time
from time import perf_counter
from air_hockey_ros.simulation import Simulation
from air_hockey_ros.registration import get_team_policy, get_simulation

from air_hockey_ros.srv import StartGame
from air_hockey_ros.msg import AgentCommand, WorldState, GameResult

import importlib
import pkgutil

def _import_all_modules(package_name: str, suffix: str):
    """
    Import only modules under `package_name` whose filename (module base) ends with `suffix`.
    Example: suffix="_team_policy" will import .../simple_team_policy.py but skip setup.py, tests, etc.
    """
    pkg = importlib.import_module(package_name)
    for _finder, modname, ispkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if ispkg:
            continue
        base = modname.rsplit(".", 1)[-1]
        if not base.endswith(suffix) or base.startswith("_"):
            continue
        importlib.import_module(modname)

TERMINATE_SLEEP_TIME = 0.05
TERMINATE_TIMEOUT = 3.0
GAME_TICK_HZ = 60
DEFAULT_GAME_DURATION = 300
IS_SYS_WIN = sys.platform.startswith("win")
CREATE_NEW_PROCESS_GROUP_WIN_SYS = 0x00000200

QOS_PROFILE = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

def _terminate_proc(proc, timeout=TERMINATE_TIMEOUT):
    try:
        if proc.poll() is not None:
            return
        if IS_SYS_WIN:
            # Send CTRL_BREAK to the new process group
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            # Signal the whole group
            os.killpg(proc.pid, signal.SIGINT)
        t0 = time.time()
        while proc.poll() is None and (time.time() - t0) < timeout:
            time.sleep(TERMINATE_SLEEP_TIME)
        if proc.poll() is None:
            # Escalate
            if IS_SYS_WIN:
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

def stop_all_agents(agent_procs: List[subprocess.Popen]):
    for p in agent_procs:
        _terminate_proc(p)
    agent_procs.clear()

def create_agent_processes(agents_args: List[List[str]]):
    if IS_SYS_WIN:
        return [subprocess.Popen(args, creationflags=CREATE_NEW_PROCESS_GROUP_WIN_SYS) for args in agents_args]
    return [subprocess.Popen(args, preexec_fn=os.setsid) for args in agents_args]

def command_to_tuple(cmd):
    return (cmd.agent_id, cmd.vx, cmd.vy)

class GameManagerNode(Node):
    def __init__(self, simulation_name: str, **params):
        super().__init__('game_manager')
        self.rclpy_clock = rclpy.clock.Clock()
        self.logger = logging.getLogger('game_manager')
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:  # avoid duplicate handlers on reload
            fh = logging.FileHandler(f"game_manager.log")
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

        self.declare_parameter('tick_hz', GAME_TICK_HZ)

        self.agent_policy_paths: Dict[int, str] = {}
        self.agent_procs: List[subprocess.Popen] = []
        self.agent_commands: Dict[int, AgentCommand] = {}
        self.agent_team_map: Dict[int, Literal['A', 'B']] = {}

        try:
            self.sim: Simulation = get_simulation(simulation_name)(**params)
        except Exception as e:
            self.logger.error("Simulation name doesn't exist")
            self.logger.error(e)
            raise e

        self.game_active: bool = False

        # Publishers / Subscribers
        self.agent_command_sub = self.create_subscription(
            AgentCommand,
            '/agent_command',
            self._agent_command_cb,
            QOS_PROFILE
        )
        self.agent_a_update_pub = self.create_publisher(
            WorldState,
            '/world_update_a',
            QOS_PROFILE
        )

        self.agent_b_update_pub = self.create_publisher(
            WorldState,
            '/world_update_b',
            QOS_PROFILE
        )

        self.result_pub = self.create_publisher(GameResult, '/game_result', 10)

        # Service to start/init game
        self.start_srv = self.create_service(
            StartGame,
            'start_game',
            self._start_game_callback
        )

        hz = self.get_parameter('tick_hz').value
        period = 1.0 / hz

        self.timer = self.create_timer(period, self._tick)
        self._tick_lock = threading.Lock()

        self.logger.info(f"GameManagerNode initialized with {simulation_name} simulation")

    def _agent_command_cb(self, msg: AgentCommand):
        self.agent_commands[msg.agent_id] = msg

    def _start_game_callback(self, request, response):
        if self.game_active:
            response.success = False
            response.message = "A game is already active"
            response.team_a_agent_ids = []
            response.team_b_agent_ids = []
            return response

        # instantiate team managers with provided policy names
        scenario_name = request.scenario_name
        team_a_name = request.team_a_name
        team_b_name = request.team_b_name
        num_agents_team_a = request.num_agents_team_a
        num_agents_team_b = request.num_agents_team_b
        rules = json.loads(request.rules)
        self.game_duration = rules.get('game_duration', DEFAULT_GAME_DURATION)

        self.logger.info(f"Starting game: {scenario_name} - "
                         + f"{team_a_name}: {num_agents_team_a} vs {team_b_name}: {num_agents_team_b}")

        self.agent_commands.clear()
        team_a_agent = get_team_policy(team_a_name)(num_agents_team_a=num_agents_team_a,
                                                    num_agents_team_b=num_agents_team_b,
                                                    team='A', **rules)
        team_a_policies = team_a_agent.get_policies()

        team_b_agent = get_team_policy(team_b_name)(num_agents_team_a=num_agents_team_a,
                                                    num_agents_team_b=num_agents_team_b,
                                                    team='B', **rules)
        team_b_policies = team_b_agent.get_policies()

        agents_a = [i for i in range(num_agents_team_a)]
        agents_b = [i for i in range(num_agents_team_a, num_agents_team_a + num_agents_team_b)]

        for aid in agents_a:
            self.agent_team_map[aid] = 'A'
        for aid in agents_b:
            self.agent_team_map[aid] = 'B'

        # Serialize agent policies and store paths
        self._policy_dir = tempfile.mkdtemp(prefix="ahg_policies_")
        for aid, policy in zip(agents_a + agents_b, team_a_policies + team_b_policies):
            policy_path = os.path.join(self._policy_dir, f"policy_{aid}.pkl")
            with open(policy_path, "wb") as f:
                pickle.dump(policy, f)
            self.agent_policy_paths[aid] = policy_path

        # Create agent args
        agents_args = [[
            "ros2", "run", "air_hockey_ros", "agent_node.py",
            "--agent_id", str(aid),
            "--team", "a", "--log", "True",
            "--policy_path", self.agent_policy_paths[aid]
        ] for aid in agents_a]
        agents_args.extend([[
            "ros2", "run", "air_hockey_ros", "agent_node.py",
            "--agent_id", str(aid),
            "--team", "b", "--log", "True",
            "--policy_path", self.agent_policy_paths[aid]
        ] for aid in agents_b])

        # Launch agents
        self.agent_procs = create_agent_processes(agents_args)

        # inform simulation about new game
        self.sim.reset_game(num_agents_team_a=num_agents_team_a, num_agents_team_b=num_agents_team_b)
        self.sim_width, self.sim_height = self.sim.get_table_sizes()

        response.success = True
        response.message = "Game started"
        response.team_a_agent_ids = agents_a
        response.team_b_agent_ids = agents_b

        self.game_start_time = perf_counter()
        self.game_active = True
        return response

    def end_game(self):
        self.game_active = False
        with self._tick_lock:
            pass
        scores = self.sim.end_game()
        stop_all_agents(self.agent_procs)
        shutil.rmtree(self._policy_dir, ignore_errors=True)

        team_a_score = scores['team_a_score']
        team_b_score = scores['team_b_score']
        if team_a_score > team_b_score:
            winner = "Team A"
        elif team_b_score > team_a_score:
            winner = "Team B"
        else:
            winner = "Draw"
        self.logger.info(f"Ending game: Team a: {team_a_score}, Team b: {team_b_score}, Winner: {winner}")
        self.result_pub.publish(GameResult(team_a_score=team_a_score, team_b_score=team_b_score, winner=winner))

    def _tick(self):
        if not self.game_active:
            return

        call_end = False
        with self._tick_lock:
            # Apply agent commands to the simulation
            commands = [command_to_tuple(cmd) for cmd in self.agent_commands.values()]
            self.agent_commands.clear()
            self.sim.apply_commands(commands)

            # Get the updated state from the simulation
            world_state = self.sim.get_world_state()

            self.logger.info(world_state)
            state_a, state_b = self.transform_world_state(world_state)

            self.agent_a_update_pub.publish(state_a)
            self.agent_b_update_pub.publish(state_b)

            if perf_counter() - self.game_start_time >= self.game_duration:
                call_end = True

        if call_end:
            self.end_game()

    def transform_world_state(self, world_state: Dict) -> Tuple[WorldState, WorldState]:
        """
        Transform the world state into fitting world states for both teams.
        """
        time_stamp = self.rclpy_clock.now().to_msg()
        state_a = WorldState(stamp=time_stamp, **world_state)
        state_b = WorldState(stamp=time_stamp,
                             puck_x=self.sim_width - world_state['puck_x'],
                             puck_y=self.sim_height - world_state['puck_y'],
                             puck_vx=-world_state['puck_vx'],
                             puck_vy=-world_state['puck_vy'],
                             agent_x=[self.sim_width - x for x in world_state['agent_x']],
                             agent_y=[self.sim_height - y for y in world_state['agent_y']],
                             agent_vx=[-vx for vx in world_state['agent_vx']],
                             agent_vy=[-vy for vy in world_state['agent_vy']], )
        return state_a, state_b

def main(args=None):
    rclpy.init(args=args)

    _import_all_modules("policies", suffix="team_policy")
    _import_all_modules("simulations", suffix="simulation")

    sim_name = sys.argv[1]
    sim_params = {}

    for arg in sys.argv[2:]:
        if "::=" in arg and not arg.startswith("_"):
            k, v = arg.split("::=", 1)
            # basic type conversion
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            elif v.isdigit():
                v = int(v)
            else:
                try:
                    v = float(v)
                except ValueError:
                    pass  # keep as string
            sim_params[k] = v

    node = GameManagerNode(simulation_name=sim_name, **sim_params)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
