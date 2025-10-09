# ros_mock.py
from __future__ import annotations
from typing import Dict, List, Tuple, Callable, Iterable, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import math
import os
import time

# --- Expected minimal sim API (adapted from your nodes) ---
# sim.reset_game(num_agents_team_a, num_agents_team_b, **rules)
# sim.apply_commands(commands: List[Tuple[agent_id, vx, vy]])
# sim.get_world_state() -> Dict with keys similar to node code:
#   team_a_score, team_b_score, puck_x, puck_y, puck_vx, puck_vy, agent_x (list), agent_y (list)
# sim.get_table_sizes() -> (width, height)

class RosMock:
    """
    Minimal orchestrator that does what the ROS nodes do:
      - Build team-specific views of world state
      - Ask each policy for its (vx, vy)
      - Apply all commands to the simulation
    No stamps, no services, no timers.
    """

    def __init__(
        self,
        sim,
        parallel: Optional[str] = "thread",  # 'thread' | 'process' | None
        max_workers: Optional[int] = None,
    ):
        self.sim = sim
        self.reverse_action_map = {}  # agent_id -> bool
        if parallel not in (None, "thread", "process"):
            raise ValueError("parallel must be None, 'thread', or 'process'")
        self.parallel = parallel
        self.max_workers = max_workers

    def close(self):
        self.sim.end_game()
        for p in (self.team_a_policies + self.team_b_policies):
            p.on_agent_close()


    # ---------- Team views (mirrors what GameManagerNode does) ----------
    def _transform_for_team_b(self, ws: Dict) -> Dict:
        # Based on transform_world_state mirroring in game_manager_node.py (no stamps). :contentReference[oaicite:5]{index=5}
        return dict(
            team_a_score=ws['team_a_score'],
            team_b_score=ws['team_b_score'],
            puck_x=self.width - ws['puck_x'],
            puck_y=self.height - ws['puck_y'],
            puck_vx=-ws['puck_vx'],
            puck_vy=-ws['puck_vy'],
            agent_x=[self.width - x for x in ws['agent_x']],
            agent_y=[self.height - y for y in ws['agent_y']],
        )

    def _world_state_for_team(self, ws: Dict) -> Tuple[Dict, Dict]:
        # A: identity, B: mirrored (no stamp fields)
        return ws, self._transform_for_team_b(ws)

    # ---------- Public API ----------
    def reset(self,
        num_agents_team_a: int,
        num_agents_team_b: int,
        team_a_policy_class,
        team_b_policy_class,
        **rules):
        team_a_policy = team_a_policy_class(num_agents_team_a=num_agents_team_a,
                                            num_agents_team_b=num_agents_team_b,
                                            team='A', **rules)
        team_b_policy = team_b_policy_class(num_agents_team_a=num_agents_team_a,
                                            num_agents_team_b=num_agents_team_b,
                                            team='B', **rules)
        self.team_a_policies = team_a_policy.get_policies()
        self.team_b_policies = team_b_policy.get_policies()
        self.num_agents_team_a = num_agents_team_a
        self.num_agents_team_b = num_agents_team_b
        # Ensure sim is aware of agent counts and geometry
        self.sim.reset_game(num_agents_team_a, num_agents_team_b, **rules)
        self.width, self.height = self.sim.get_table_sizes()

        # Agent id space: A = [0..A-1], B = [A..A+B-1]
        for aid in range(self.num_agents_team_a):
            self.reverse_action_map[aid] = False
        for j in range(self.num_agents_team_b):
            self.reverse_action_map[self.num_agents_team_a + j] = True  # mirror team B like in node

        # Call optional init hooks
        for p in (self.team_a_policies + self.team_b_policies):
            p.on_agent_init()

    def step(self) -> Dict:
        """
        One “tick”:
          1) Read world_state from sim.
          2) Build team-specific views.
          3) Get commands from policies (parallelizable).
          4) Apply all commands to sim.
          5) Return the NEW world state (post-apply).
        """
        ws = self.sim.get_world_state()
        ws_a, ws_b = self._world_state_for_team(ws)

        # Query policies
        cmds: List[Tuple[int, float, float]] = []
        if self.parallel is None:
            # Sequential
            for i, pol in enumerate(self.team_a_policies):
                vx, vy = pol.update(ws_a)  # like agent_node uses policy.update(world_state_to_dict) :contentReference[oaicite:7]{index=7}
                cmds.append(i, vx, vy)
            for j, pol in enumerate(self.team_b_policies):
                vx, vy = pol.update(ws_b)
                aid = self.num_agents_team_a + j
                cmds.append((aid, -vx, -vy))
        else:
            Exec = ThreadPoolExecutor if self.parallel == "thread" else ProcessPoolExecutor

            def _call_update(pol, state, aid, flip):
                vx, vy = pol.update(state)
                if flip:
                    return aid, -vx, vy
                return aid, vx, vy

            with Exec(max_workers=self.max_workers) as ex:
                jobs = []
                for i, pol in enumerate(self.team_a_policies):
                    jobs.append((pol, ws_a, i, False))
                for j, pol in enumerate(self.team_b_policies):
                    jobs.append((pol, ws_b, self.num_agents_team_a + j, True))

                for aid, vx, vy in ex.map(lambda args: _call_update(*args), jobs):
                    cmds.append((aid, vx, vy))

        # Apply to sim and return the new state
        self.sim.apply_commands(cmds)
        return self.sim.get_world_state()
