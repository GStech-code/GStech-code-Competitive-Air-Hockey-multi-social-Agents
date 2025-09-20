from typing import List, Dict, Tuple
import math
from air_hockey_ros import Simulation, register_simulation

@register_simulation('mock')
class MockSimulation(Simulation):
    def __init__(self, width=800, height=600, step_size=8.0, use_physics=False):
        super().__init__(width=width, height=height, step_size=step_size)
        self.use_physics = use_physics

        # Puck
        self.puck_x = width / 2.0
        self.puck_y = height / 2.0
        self.puck_vx = 0.0
        self.puck_vy = 0.0

        # Agents (all together in one list)
        self.agent_x = []
        self.agent_y = []
        self.agent_vx = []
        self.agent_vy = []

    def reset_game(self, num_agents_team_a, num_agents_team_b, **params):
        super().reset_game(num_agents_team_a, num_agents_team_b)

        self.width = params.get("width", self.width)
        self.height = params.get("height", self.height)
        self.step_size = float(params.get("step_size", self.step_size))

        if 'agent_positions' in params:
            for i, pos in enumerate(params["agent_positions"]):
                self.agent_x[i] = pos[0]
                self.agent_y[i] = pos[1]
        else:
            gap_a = self.height / (self.num_agents_team_a + 1)
            agent_a_y = [gap_a * (i + 1) for i in range(self.num_agents_team_a)]
            gap_b = self.height / (self.num_agents_team_b + 1)
            agent_b_y = [self.height - gap_b * (i + 1) for i in range(self.num_agents_team_b)]
            self.agent_y = agent_a_y + agent_b_y
            agent_a_x = [self.width / 4.0] * self.num_agents_team_a
            agent_b_x = [3 * self.width / 4.0] * self.num_agents_team_b
            self.agent_x = agent_a_x + agent_b_x

        self.agent_vx = [0.0] * self.num_agents
        self.agent_vy = [0.0] * self.num_agents

        self.team_a_score = 0
        self.team_b_score = 0

    def apply_commands(self, commands: List[Tuple]):
        """
        commands: dict[int, tuple[int,int]]
        keys = agent index (0..N-1), values = (vx, vy)
        """
        for command in commands:
            id, vx, vy = command
            vx = float(max(-2, min(2, vx)))
            vy = float(max(-2, min(2, vy)))
            self.agent_vx[id] = vx
            self.agent_vy[id] = vy

            if self.use_physics:
                nx = self.agent_x[id] + vx * self.step_size
                ny = self.agent_y[id] + vy * self.step_size
                # Clamp inside field
                self.agent_x[id] = min(max(20, nx), self.width - 20)
                self.agent_y[id] = min(max(20, ny), self.height - 20)

    def get_world_state(self) -> Dict:
        return {
            "puck_x": self.puck_x, "puck_y": self.puck_y,
            "puck_vx": self.puck_vx, "puck_vy": self.puck_vy,
            "agent_x": self.agent_x[:],
            "agent_y": self.agent_y[:],
            "agent_vx": self.agent_vx[:],
            "agent_vy": self.agent_vy[:],
        }
