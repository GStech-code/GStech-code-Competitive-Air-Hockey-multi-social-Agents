# uct_planner.py
# TODO: Keep changing
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math
import random

from .uct_heuristic import UCTHeuristic
from short_term_objectives import ObjectiveEnum
from .ros_mock import RosMock

WorldState = Dict
AgentID = int

# Candidate objective combos provider for a team’s LS agent IDs (no ws here by design).
CombosFn = Callable[[bool], Sequence[Tuple[ObjectiveEnum, ...]]]

# Map chosen combo -> {agent_id: objective} (LS via combo; non-LS forced PASS_SHOT).
# We pass is_team_a to avoid guessing by ids.
AssignFn = Callable[[WorldState, bool, Sequence[AgentID], Tuple[ObjectiveEnum, ...]], Dict[AgentID, ObjectiveEnum]]


@dataclass
class Node:
    ws_count: int
    parent: Optional["Node"] = None
    children: Dict[Tuple[int, int], "Node"] = field(default_factory=dict)  # (a_idx, b_idx) -> child
    N: int = 0
    edge_N: Dict[Tuple[int, int], int] = field(default_factory=dict)
    edge_W: Dict[Tuple[int, int], float] = field(default_factory=dict)
    a_combo_idx: Optional[List[int]] = None
    b_combo_idx: Optional[List[int]] = None
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
        number_of_planning_steps: int,  # steps per cycle block
        ros_mock: RosMock,
        heuristic: UCTHeuristic,
        combos_fn: CombosFn,
        assign_fn: AssignFn,
        policy_matrix: Dict[AgentID, Dict[ObjectiveEnum, object]],
        team_a_agents: Sequence[AgentID],
        team_b_agents: Sequence[AgentID],
        exploration_c: float = 1.41421356237,
        rng: Optional[random.Random] = None,
    ):
        self.block = int(number_of_planning_steps)
        assert self.block >= 1
        self.ros_mock = ros_mock
        self.heuristic_obj = heuristic
        self._h = None  # heuristic(ws) relative to cycle start
        self.combos_fn = combos_fn
        self.assign_fn = assign_fn
        self.P = policy_matrix
        self.teamA = list(team_a_agents)
        self.teamB = list(team_b_agents)
        self.c = float(exploration_c)
        self.rng = rng or random.Random()

        self._root: Optional[Node] = None
        self._root_ws: Optional[WorldState] = None
        self._root_steps_left: int = self.block
        self._last_root_choice: Optional[Tuple[int, int]] = None
        self._last_root_assigns: Optional[
            Tuple[Dict[AgentID, ObjectiveEnum], Dict[AgentID, ObjectiveEnum]]
        ] = None

    # ---- lifecycle ----
    def reset(self) -> None:
        self._root = None
        self._root_ws = None
        self._h = None
        self._last_root_choice = None
        self._last_root_assigns = None

    def _set_heuristic_baseline(self, ws: WorldState) -> None:
        a0 = int(ws["team_a_score"])
        b0 = int(ws["team_b_score"])
        self._h = lambda w: self.heuristic_obj.evaluate(a0, b0, w)

    def start_cycle(self, ws: WorldState, *, ws_count: int, steps_left_in_cycle: int) -> None:
        self._root_ws = ws
        self._root_steps_left = int(steps_left_in_cycle)
        self._root = Node(ws_count=ws_count)
        self._last_root_choice = None
        self._last_root_assigns = None
        self._set_heuristic_baseline(ws)

    def update_world_state(self, ws: WorldState, *, ws_count: int, steps_left_in_cycle: int) -> None:
        self._root_steps_left = int(steps_left_in_cycle)

        if not self._root:
            self.start_cycle(ws, ws_count=ws_count, steps_left_in_cycle=steps_left_in_cycle)
            return

        if self._root.ws_count == ws_count:
            self._root_ws = ws
            return

        # try to reuse a direct child that already corresponds to this ws_count
        for child in self._root.children.values():
            if child.ws_count == ws_count:
                child.parent = None
                self._root = child
                self._root_ws = child.cached_ws if child.cached_ws is not None else ws
                self._last_root_choice = None
                self._last_root_assigns = None
                self._set_heuristic_baseline(self._root_ws)
                return

        # otherwise just refresh ws & baseline
        self._root_ws = ws
        self._set_heuristic_baseline(ws)

    # ---- planning ----
    # uct_planner.py  — add:
    def expand_once(self) -> Tuple[Dict[int, ObjectiveEnum], Dict[int, ObjectiveEnum]]:
        if not self._root or self._root_ws is None or self._h is None:
            raise RuntimeError("start_cycle/update_world_state before expand_once().")

        # exactly ONE UCT expansion
        self._search_once(self._root, self._root_ws, is_root=True)

        # choose best edge by exploitation
        best_edge, _ = self._best_child_edge(self._root, prefer_exploitation=True)
        if best_edge is None:
            # fallback to first legal
            a_combos, b_combos = self._get_node_combos(self._root)
            if not a_combos or not b_combos:
                self._last_root_choice = None
                self._last_root_assigns = ({}, {})
                return {}, {}
            best_edge = (0, 0)

        ai, bi = best_edge
        a_combos, b_combos = self._get_node_combos(self._root)
        a_assign = self.assign_fn(self._root_ws, True, self.teamA, a_combos[ai])
        b_assign = self.assign_fn(self._root_ws, False, self.teamB, b_combos[bi])

        self._last_root_choice = best_edge
        self._last_root_assigns = (a_assign, b_assign)
        return a_assign, b_assign

    def commit_root_choice(self) -> None:
        """Advance root into the chosen child so its subtree is kept for the next cycle."""
        if not self._root or self._root_ws is None:
            return
        if self._last_root_choice is None or self._last_root_assigns is None:
            return

        jk = self._last_root_choice
        child = self._root.children.get(jk)

        if child is None:
            a_combos, b_combos = self._get_node_combos(self._root)
            ai, bi = jk
            a_assign, b_assign = self._last_root_assigns
            steps = max(0, self.block - self._root_steps_left)
            if steps == 0:
                next_ws = self._root_ws
                used = 0
            else:
                next_ws = self._rollout(self._root_ws, a_assign, b_assign, steps)
                used = steps
            child = Node(
                ws_count=self._root.ws_count + used,
                parent=self._root,
                cached_ws=next_ws
            )
            self._root.children[jk] = child

        child.parent = None
        self._root = child
        if child.cached_ws is not None:
            self._root_ws = child.cached_ws
        self._last_root_choice = None
        self._last_root_assigns = None
        self._set_heuristic_baseline(self._root_ws)

    # ---- core UCT ----
    def _search_once(self, node: Node, ws: WorldState, *, is_root: bool) -> float:
        a_combos, b_combos = self._get_node_combos(node)
        if not a_combos or not b_combos:
            v = self._h(ws) if self._h else 0.0
            node.N += 1
            return v

        jk = self._select_edge_ucb(node)
        ai, bi = jk

        a_assign = self.assign_fn(ws, True,  self.teamA, a_combos[ai])
        b_assign = self.assign_fn(ws, False, self.teamB, b_combos[bi])

        steps = self._root_steps_left if is_root else self.block
        next_ws = self._rollout(ws, a_assign, b_assign, steps)
        next_count = node.ws_count + steps

        child = node.children.get(jk)
        if child is None:
            child = Node(ws_count=next_count, parent=node, cached_ws=next_ws)
            node.children[jk] = child

        v = self._search_once(child, next_ws, is_root=False)

        node.N += 1
        node.edge_N[jk] = node.edge_N.get(jk, 0) + 1
        node.edge_W[jk] = node.edge_W.get(jk, 0.0) + v
        return v

    def _get_node_combos(
        self, node: Node
    ) -> Tuple[List[Tuple[ObjectiveEnum, ...]], List[Tuple[ObjectiveEnum, ...]]]:
        if node.a_combo_idx is not None and node.b_combo_idx is not None:
            a_all = list(self.combos_fn(True))
            b_all = list(self.combos_fn(False))
            A = [a_all[i] for i in node.a_combo_idx if i < len(a_all)]
            B = [b_all[i] for i in node.b_combo_idx if i < len(b_all)]
            return A, B

        a_all = list(self.combos_fn(True))
        b_all = list(self.combos_fn(False))

        node.a_combo_idx = list(range(len(a_all)))
        node.b_combo_idx = list(range(len(b_all)))
        return a_all, b_all

    def _select_edge_ucb(self, node: Node) -> Tuple[int, int]:
        a_len = len(node.a_combo_idx or [])
        b_len = len(node.b_combo_idx or [])
        if a_len == 0 or b_len == 0:
            return (0, 0)

        cands: List[Tuple[int, int]] = [(i, j) for i in range(a_len) for j in range(b_len)]

        # expand unvisited first
        for jk in cands:
            if jk not in node.edge_N:
                return jk

        best_jk = cands[0]
        best_score = -float("inf")
        for jk in cands:
            n = node.edge_N[jk]
            q = (node.edge_W[jk] / n) if n > 0 else 0.0
            u = self.c * math.sqrt(math.log(max(1, node.N)) / n)
            s = q + u
            if s > best_score:
                best_score = s
                best_jk = jk
        return best_jk

    def _best_child_edge(self, node: Node, *, prefer_exploitation: bool
                        ) -> Tuple[Optional[Tuple[int, int]], float]:
        if not node.edge_N:
            return None, float("nan")
        if prefer_exploitation:
            best = None
            bestN = -1
            bestQ = -float("inf")
            for jk, n in node.edge_N.items():
                q = node.edge_W[jk] / n
                if (n > bestN) or (n == bestN and q > bestQ):
                    best = jk
                    bestN = n
                    bestQ = q
            return best, bestQ
        else:
            best = None
            bestQ = -float("inf")
            for jk, n in node.edge_N.items():
                q = node.edge_W[jk] / n
                if q > bestQ:
                    best = jk
                    bestQ = q
            return best, bestQ

    # ---- rollout via RosMock + ShortAgentPolicies ----
    def _rollout(self, ws: WorldState,
                 a_assign: Dict[AgentID, ObjectiveEnum],
                 b_assign: Dict[AgentID, ObjectiveEnum],
                 steps: int) -> WorldState:
        self.ros_mock.load_world_state(ws)

        a_pols: List[object] = []
        for aid in self.teamA:
            obj = a_assign[aid]
            a_pols.append(self.P[aid][obj])

        b_pols: List[object] = []
        for bid in self.teamB:
            obj = b_assign[bid]
            b_pols.append(self.P[bid][obj])

        self.ros_mock.apply_agent_policies(a_pols, b_pols)
        for _ in range(max(0, steps)):
            self.ros_mock.step()

        return self.ros_mock.get_world_state()
