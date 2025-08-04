import subprocess
import pickle
import tempfile, shutil
import os
import rclpy
from rclpy.node import Node
from typing import Dict, List, Literal, Tuple

from air_hockey_ros.srv import StartGame
from air_hockey_ros.msg import AgentCommand, WorldState

from policy import TeamPolicy, get_team_policy
from simulation import SimulationInterface


class GameManagerNode(Node):
    def __init__(self, simulationInterfaceClass: type[SimulationInterface]):
        super().__init__('game_manager')
        self.declare_parameter('tick_hz', 10)

        self.agent_policy_paths: Dict[str, str] = {}
        self.agent_procs: Dict[str, subprocess.Popen] = {}
        self.agent_commands: Dict[str, AgentCommand] = {}
        self.agent_team_map: Dict[str, Literal['A', 'B']] = {}

        self.sim: SimulationInterface = simulationInterfaceClass(self.get_logger())  # inject real sim

        self.game_active: bool = False

        # Publishers / Subscribers
        self.agent_command_sub = self.create_subscription(
            AgentCommand,
            '/agent_command',
            self._agent_command_cb,
            100
        )
        self.agent_a_update_pub = self.create_publisher(
            WorldState,
            '/world_update_a',
            100
        )

        self.agent_b_update_pub = self.create_publisher(
            WorldState,
            '/world_update_b',
            100
        )

        # Service to start/init game
        self.start_srv = self.create_service(
            StartGame,
            'start_game',
            self._start_game_callback
        )

        hz = self.get_parameter('tick_hz').get_parameter_value().double_value
        period = 1.0 / hz
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info("GameManagerNode initialized.")

    def _agent_command_cb(self, msg: AgentCommand):
        self.agent_commands[msg.agent_id] = msg

    def _clear_agent_commands(self):
        self.agent_commands.clear()

    def _start_game_callback(self, request, response):
        # instantiate team managers with provided policy names
        scenario_name = request.scenario_name
        team_a_name = request.team_a_name
        team_b_name = request.team_b_name
        num_agents_team_a = request.num_agents_team_a
        num_agents_team_b = request.num_agents_team_b
        rules = request.rules


        self.get_logger().info(f"Starting game: {scenario_name} - "
                               + f"{team_a_name}: {num_agents_team_a} vs {team_b_name}: {num_agents_team_b}")

        team_a_policies = get_team_policy(team_a_name)(num_agents_team_a).get_policies()
        team_b_policies = get_team_policy(team_b_name)(num_agents_team_b).get_policies()

        agents_a = [str(i) for i in range(num_agents_team_a)]
        agents_b = [str(i) for i in range(num_agents_team_a, num_agents_team_a + num_agents_team_b)]

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

        # inform simulation about new game
        self.sim.reset_game(**{
            'team_a_agents': agents_a,
            'team_b_agents': agents_b,
            'rules': rules
        })

        response.success = True
        response.message = "Game started"
        response.team_a_agent_ids = agents_a
        response.team_b_agent_ids = agents_b

        # Launch agents
        for aid in agents_a:
            proc = subprocess.Popen([
                "ros2", "run", "air_hockey_game", "agent_node",
                "--agent_id", aid,
                "--team", "A",
                "--policy_path", self.agent_policy_paths[aid]
            ])
            self.agent_procs[aid] = proc

        for aid in agents_b:
            proc = subprocess.Popen([
                "ros2", "run", "air_hockey_game", "agent_node",
                "--agent_id", aid,
                "--team", "B",
                "--policy_path", self.agent_policy_paths[aid]
            ])
            self.agent_procs[aid] = proc

        self.game_active = True
        return response

    def end_game(self):
        self.game_active = False
        scores = self.sim.end_game()
        shutil.rmtree(self._policy_dir, ignore_errors=True)
        return scores

    def _tick(self):
        if not self.game_active:
            return

        # Apply agent commands to the simulation
        commands = list(self.agent_commands.values())
        self._clear_agent_commands()
        self.sim.apply_commands(commands)

        # Get the updated state from the simulation
        world_state = self.sim.get_world_state()
        state_a, state_b = self.transform_world_state(world_state)

        self.agent_a_update_pub.publish(state_a)
        self.agent_b_update_pub.publish(state_b)

    def transform_world_state(self, world_state) -> Tuple[WorldState, WorldState]:
        """
        Transform the world state into fitting world states for both teams.
        """
        return world_state, world_state

