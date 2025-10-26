# long_term_planner.py
from typing import Dict, Tuple, List
from .bus import Mailbox
from .short_policy_chooser import ShortPolicyChooser
from .short_agent_policy import ShortAgentPolicy
from .uct_planning import RosMock, UCTHeuristic, UCTPlanner
from simulations import BaseSimulation
from .short_term_objectives import ObjectiveEnum, OBJECTIVE_COMBINATIONS

NUMBER_OF_PLANNING_STEPS = 4
MAX_LS_AGENTS = 4  # only first 4 are long-short; others are PASS_SHOT

# Internal tuning: how much UCT thinking to do per call (can be time budgets if you prefer)
_ITER_ON_NEW_WS   = 96   # each update (once)
_ITER_ON_SAME_WS  = 64   # optional extra thinking calls (0..many per update)

class LongTermPlanner:
    """
    Scheduling model you described:
      • new_ws_step() — called exactly once per update:
          - refresh UCT with the new WS
          - run UCT thinking for this update
          - decrement cycle counter
          - when counter hits 0: publish instruction and commit subtree; reset counter
      • step() — may be called 0..many times between updates:
          - does *extra* UCT thinking only (no counter change, no publishing)
    """
    def __init__(self, agent_id: int, team: str, num_agents_team_a: int, num_agents_team_b: int,
                 mailbox: Mailbox,
                 team_a_policies_chooser: ShortPolicyChooser,
                 team_b_policies_chooser: ShortPolicyChooser,
                 team_a_pass_policies: Dict[int, ShortAgentPolicy],
                 team_b_pass_policies: Dict[int, ShortAgentPolicy],
                 rules: Dict):

        self.mailbox = mailbox
        self.rules = dict(rules or {})
        self.rules['stuck_window'] = self.rules.get('stuck_window', 2)
        self.rules['hold_last_ticks'] = self.rules.get('hold_last_ticks', 0)

        # sim & mock
        self.ros_mock = RosMock(BaseSimulation())
        self.ros_mock.start(num_agents_team_a=num_agents_team_a,
                            num_agents_team_b=num_agents_team_b,
                            **self.rules)

        self.teamA: List[int] = [aid for aid in range(num_agents_team_a)]
        self.teamB: List[int] = [aid for aid in range(num_agents_team_a, num_agents_team_a + num_agents_team_b)]
        self.lsA = min(MAX_LS_AGENTS, num_agents_team_a)
        self.lsB = min(MAX_LS_AGENTS, num_agents_team_b)

        # choosers
        self.team_a_chooser = team_a_policies_chooser
        self.team_b_chooser = team_b_policies_chooser

        # policy matrix (use chooser.get_policies())
        a_map = self.team_a_chooser.get_policies()
        b_map = self.team_b_chooser.get_policies()

        self.policy_matrix: Dict[int, Dict[ObjectiveEnum, ShortAgentPolicy]] = {}
        for aid in self.teamA[:self.lsA]:
            self.policy_matrix[aid] = {
                ObjectiveEnum.DEFEND_LINE: a_map[(aid, ObjectiveEnum.DEFEND_LINE)],
                ObjectiveEnum.INTERCEPT:   a_map[(aid, ObjectiveEnum.INTERCEPT)],
                ObjectiveEnum.PASS_SHOT:   a_map[(aid, ObjectiveEnum.PASS_SHOT)],
                ObjectiveEnum.FAST_SHOT:   a_map[(aid, ObjectiveEnum.FAST_SHOT)],
            }
        for aid in self.teamA[self.lsA:]:
            self.policy_matrix[aid] = {ObjectiveEnum.PASS_SHOT: team_a_pass_policies[aid]}

        for bid in self.teamB[:self.lsB]:
            self.policy_matrix[bid] = {
                ObjectiveEnum.DEFEND_LINE: b_map[(bid, ObjectiveEnum.DEFEND_LINE)],
                ObjectiveEnum.INTERCEPT:   b_map[(bid, ObjectiveEnum.INTERCEPT)],
                ObjectiveEnum.PASS_SHOT:   b_map[(bid, ObjectiveEnum.PASS_SHOT)],
                ObjectiveEnum.FAST_SHOT:   b_map[(bid, ObjectiveEnum.FAST_SHOT)],
            }
        for bid in self.teamB[self.lsB:]:
            self.policy_matrix[bid] = {ObjectiveEnum.PASS_SHOT: team_b_pass_policies[bid]}

        # heuristic
        self.uct_heuristic = UCTHeuristic(team_a_agent_ids=self.teamA,
                                          team_b_agent_ids=self.teamB,
                                          **self.rules)

        team_a_combinations = OBJECTIVE_COMBINATIONS.get(self.lsA, [])
        team_b_combinations = OBJECTIVE_COMBINATIONS.get(self.lsB, [])
        def assign_fn(ws: Dict, is_team_a: bool, agent_ids: List[int], combo: Tuple[ObjectiveEnum, ...]):
            assign: Dict[int, ObjectiveEnum] = {}

            agents_x: List[float] = ws["agent_x"]
            chooser = self.team_a_chooser if is_team_a else self.team_b_chooser
            ls = self.lsA if is_team_a else self.lsB
            ls_ids = list(agent_ids[:ls])
            tail_ids = list(agent_ids[ls:])

            if ls_ids and combo:
                # chooser returns mapping for all its known agents; filter to LS ids
                full_map = chooser.choose_objectives(agents_x, list(combo))  # {aid: enum}
                for aid in ls_ids:
                    assign[aid] = full_map[aid]

            for aid in tail_ids:
                assign[aid] = ObjectiveEnum.PASS_SHOT

            return assign

        # ---- UCT instance ----
        is_team_a = team == 'A'
        self._steps_left_in_cycle: int = NUMBER_OF_PLANNING_STEPS
        self.is_root_change = False
        self.uct_planner = UCTPlanner(
            is_team_a=is_team_a,
            agent_id=agent_id,
            number_of_planning_steps=NUMBER_OF_PLANNING_STEPS,
            ros_mock=self.ros_mock,
            heuristic=self.uct_heuristic,
            team_a_combinations=team_a_combinations,
            team_b_combinations=team_b_combinations,
            assign_fn=assign_fn,
            policy_matrix=self.policy_matrix,
            team_a_agents=self.teamA,
            team_b_agents=self.teamB,
        )

    # ---- scheduler hooks ----
    def new_ws_step(self) -> None:
        ws = self.mailbox.latest_world_state.get()
        if ws is None:
            return
        # consume one tick of the cycle before using the leftover at root
        self._steps_left_in_cycle -= 1

        if self.is_root_change:
            self.uct_planner.commit_root_choice()  # reuse subtree next cycle
            self.is_root_change = False

        # keep current cycle budget for the root (first time = 32)
        self.uct_planner.update_world_state(ws)
        # exactly ONE UCT expansion this update
        self.mailbox.latest_instruction.set(self.uct_planner.expand_once())
        # publish & commit at cycle boundary
        if self._steps_left_in_cycle <= 0:
            self.mailbox.status_change_flag.inform()  # inform once per cycle
            self.is_root_change = True
            self._steps_left_in_cycle = NUMBER_OF_PLANNING_STEPS  # reset to 32


    def step(self) -> None:
        self.mailbox.latest_instruction.set(self.uct_planner.expand_once())

    def change_instruction(self, instruction):
        self.mailbox.latest_instruction.set(instruction)
        self.mailbox.status_change_flag.inform()