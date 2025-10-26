# uct_planner.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import math
import random

from .uct_heuristic import UCTHeuristic
from ..short_term_objectives import ObjectiveEnum
from ..short_agent_policy import ShortAgentPolicy
from .ros_mock import RosMock

WorldState = Dict
AgentID = int

# Map chosen combo -> {agent_id: objective} (LS via combo; non-LS forced PASS_SHOT).
# We pass is_team_a to avoid guessing by ids.
AssignFn = Callable[[WorldState, bool, List[AgentID], Tuple[ObjectiveEnum, ...]], Dict[AgentID, ObjectiveEnum]]


@dataclass
class Node:
    ws_count: int
    children: Dict[Tuple[int, int], "Node"] = field(default_factory=dict)  # (a_idx, b_idx) -> child
    N: int = 0
    edge_N: Dict[Tuple[int, int], int] = field(default_factory=dict)
    edge_W: Dict[Tuple[int, int], float] = field(default_factory=dict)
    cached_ws: Optional[WorldState] = None


class UCTPlanner:
    """
    Cycle-aware UCT:
      - Root baseline ties to score at cycle start.
      - Reuses the chosen child’s subtree across cycles.
      - Rollouts simulate 'block' steps from root; full 'block' at inner nodes.
    """

    def __init__(
            self,
            *,
            is_team_a: bool,
            agent_id: int,
            number_of_planning_steps: int,  # steps per cycle block
            ros_mock: RosMock,
            heuristic: UCTHeuristic,
            team_a_combinations: List[Tuple[ObjectiveEnum]],
            team_b_combinations: List[Tuple[ObjectiveEnum]],
            assign_fn: AssignFn,
            policy_matrix: Dict[AgentID, Dict[ObjectiveEnum, object]],
            team_a_agents: List[AgentID],
            team_b_agents: List[AgentID],
            exploration_c: float = 1.41421356237,
            diminishing_gamma: float = 0.95,  # Related to the rate of forgetting old ws heuristic scores
            rng: Optional[random.Random] = None,
    ):
        self.is_team_a = is_team_a
        self.agent_id = agent_id
        self.block = number_of_planning_steps
        self.init_block = self.block + 1  # Add 1 to negate the first subtraction
        self.ros_mock = ros_mock
        self.heuristic_obj = heuristic
        self._h = None  # heuristic(ws) relative to cycle start
        self.team_a_combinations = team_a_combinations
        self.team_b_combinations = team_b_combinations
        self.assign_fn = assign_fn
        self.P = policy_matrix
        self.teamA = team_a_agents
        self.teamB = team_b_agents
        self.c = exploration_c
        self.gamma = diminishing_gamma
        self.rng = rng or random.Random()
        self.ws_count = 0

        self._root: Optional[Node] = None
        self._root_ws: Optional[WorldState] = None
        self._root_steps_left: int = self.init_block
        self._last_root_choice: Optional[Tuple[int, int]] = None

        self.len_a_combs = len(team_a_combinations)
        self.len_b_combs = len(team_b_combinations)
        if self.len_a_combs == 0:
            self.a_cands = (-1, )
        else:
            self.a_cands = [i for i in range(self.len_a_combs)]

        if self.len_b_combs == 0:
            self.b_cands = (-1, )
        else:
            self.b_cands = [i for i in range(self.len_b_combs)]

        self.cands = [(i, j) for i in self.a_cands for j in self.b_cands]
        self.num_cands = len(self.cands)

        if self.len_a_combs:
            a_choice = 1
        else:
            a_choice = -1

        if self.len_b_combs:
            b_choice = 1
        else:
            b_choice = -1

        self._previous_root_choice: Tuple[int, int] = (a_choice, b_choice)

    # ---- lifecycle ----

    def _set_heuristic_baseline(self, ws: WorldState) -> None:
        a0 = int(ws["team_a_score"])
        b0 = int(ws["team_b_score"])
        self._h = lambda w: self.heuristic_obj.evaluate(a0, b0, w)

    def start_cycle(self, ws: WorldState) -> None:
        self._root_ws = ws
        self._root = Node(ws_count=self.ws_count, cached_ws=ws)
        self._last_root_choice = None
        self._set_heuristic_baseline(ws)

    def update_world_state(self, ws: WorldState) -> None:
        self._root_steps_left -= 1
        self.ws_count += 1

        ws = self._rollout(ws, self._previous_root_choice, self._root_steps_left)

        if not self._root:
            self.start_cycle(ws)
            return

        # otherwise just refresh ws & baseline
        self._root_ws = ws
        self._root.cached_ws = ws
        self._set_heuristic_baseline(self._root_ws)

    def commit_root_choice(self) -> None:
        """Advance root into the chosen child so its subtree is kept for the next cycle."""
        self._root_steps_left = self.init_block
        if not self._root or self._last_root_choice is None:
            return

        jk = self._last_root_choice
        child = self._root.children.get(jk)

        if child is None:
            return


        self._root = child
        self._previous_root_choice = jk
        self._last_root_choice = None

    # ---- planning ----
    def expand_once(self) -> ObjectiveEnum:
        # exactly ONE UCT expansion
        self._search_once(self._root, self._root_ws)

        # choose best edge by exploitation
        best_edge = self._select_edge_ucb(self._root, diminishing_mode=True)

        ai, bi = best_edge

        self._last_root_choice = best_edge
        if self.is_team_a:
            a_assign = self.assign_fn(self._root_ws, True, self.teamA, self.team_a_combinations[ai])
            return a_assign.get(self.agent_id, ObjectiveEnum.PASS_SHOT)
        b_assign = self.assign_fn(self._root_ws, False, self.teamB, self.team_b_combinations[bi])
        return b_assign.get(self.agent_id, ObjectiveEnum.PASS_SHOT)

    # ---- core UCT ----
    # Returns ws_score
    def _search_once(self, node: Node, ws: WorldState) -> float:
        # If was never visited, it means the node has no children. Create all of the children.
        if node.N == 0:
            K = self.num_cands
            ws_scores = 0.0
            for jk in self.cands:
                next_ws = self._rollout(ws=ws, jk=jk, steps=self.block)
                ws_score = self._h(next_ws)
                child = Node(cached_ws=next_ws, ws_count=self.ws_count)
                node.children[jk] = child
                node.edge_N[jk] = 1
                node.edge_W[jk] = ws_score
                ws_scores += ws_score
            node.N += K
            return ws_scores / K

        jk = self._select_edge_ucb(node, diminishing_mode=False)
        child = node.children[jk]
        ws_count_diff = self.ws_count - child.ws_count
        if ws_count_diff > 0:
            next_ws = self._rollout(ws=ws, jk=jk, steps=self.block)
            child.cached_ws = next_ws
            child.ws_count = self.ws_count
        else:
            next_ws = child.cached_ws

        s_child = self._search_once(child, next_ws)

        if ws_count_diff > 0:
            W = node.edge_W[jk]
            N = node.edge_N[jk]
            k = (self.gamma ** ws_count_diff)
            node.edge_W[jk] = (k * W + float(s_child)) * ((N + 1) / (k * N + 1.0))
        else:
            node.edge_W[jk] += s_child

        node.edge_N[jk] += 1
        node.N += 1

        return s_child

    def _select_edge_ucb(self, node: Node, diminishing_mode: bool) -> Tuple[int, int]:
        a_cands = self.a_cands
        b_cands = self.b_cands

        best_max_jk = (a_cands[0], b_cands[0])
        best_max_score = -float("inf")
        for a in a_cands:
            best_min_jk = (a, b_cands[0])
            best_min_score = float("inf")
            for b in b_cands:
                jk = (a, b)
                score = self._calc_ucb(node, jk, is_max=False, diminishing_mode=diminishing_mode)
                if score < best_min_score:
                    best_min_jk = jk
                    best_min_score = score

            score = self._calc_ucb(node, best_min_jk, is_max=True, diminishing_mode=diminishing_mode)
            if score > best_max_score:
                best_max_jk = best_min_jk
                best_max_score = score

        return best_max_jk

    def _calc_ucb(self, node: Node, jk: Tuple[int, int], is_max: bool, diminishing_mode: bool) -> float:
        n = node.edge_N[jk]  # N must be > 0
        q = node.edge_W[jk] / n
        u = self.c * math.sqrt(math.log(1 + node.N) / n)
        if diminishing_mode:
            diff = self.ws_count - node.children[jk].ws_count
            if diff > 0:
                q *= (self.gamma ** diff)
        if is_max:
            return q + u
        return q - u

    # ---- rollout via RosMock + ShortAgentPolicies ----
    def _rollout(self, ws: WorldState, jk, steps: int) -> WorldState:

        a_combos, b_combos = self.team_a_combinations, self.team_b_combinations
        ai, bi = jk
        a_pols: List[ShortAgentPolicy] = []
        b_pols: List[ShortAgentPolicy] = []
        if ai >= 0:
            a_assign = self.assign_fn(ws, True, self.teamA, a_combos[ai])
            for aid in self.teamA:
                objective_enum = a_assign[aid]
                a_pols.append(self.P[aid][objective_enum])

        if bi >= 0:
            b_assign = self.assign_fn(ws, False, self.teamB, b_combos[bi])
            for bid in self.teamB:
                objective_enum = b_assign[bid]
                b_pols.append(self.P[bid][objective_enum])

        self.ros_mock.load_world_state(ws)
        self.ros_mock.apply_agent_policies(a_pols, b_pols)
        for _ in range(max(0, steps)):
            self.ros_mock.step()

        return self.ros_mock.get_world_state()
