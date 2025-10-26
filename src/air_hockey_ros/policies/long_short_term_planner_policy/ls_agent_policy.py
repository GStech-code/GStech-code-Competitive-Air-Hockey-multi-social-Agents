# ls_agent_policy.py
from typing import Dict, Tuple, Optional, List
import atexit
import weakref
from .bus import Mailbox
from .scheduler import SpotlightScheduler
from .short_term_planner import ShortTermPlanner
from .short_term_objectives import ObjectiveEnum, OBJECTIVE_ENUMS
from .short_policy_chooser import ShortPolicyChooser
from .objective_producer import ObjectivesProducer
from .short_agent_policy import ShortAgentPolicy
from .long_term_planner import LongTermPlanner
from air_hockey_ros import AgentPolicy
from .types import Command

OFFENSIVE_FACTORS = [0.4, 0.5, 0.6, 0.7]

class LSAgentPolicy(AgentPolicy):
    """Pickle‑friendly: threads are created/started in on_agent_init()."""
    def __init__(self, agent_id: int,
                 team: str,
                 max_num_valid_agents: int,
                 num_agents_team_a: int,
                 num_agents_team_b: int,
                 rules: Dict,
                 starter_objective: ObjectiveEnum = ObjectiveEnum.DEFEND_LINE,
                 ):
        super().__init__(agent_id)
        self.team = team
        self.max_num_valid_agents = max_num_valid_agents
        self.num_agents_team_a = num_agents_team_a
        self.num_agents_team_b = num_agents_team_b
        self.rules = rules
        self.width = float(rules.get('width', 800))
        self.team_a_agents = [aid for aid in range(num_agents_team_a)]
        self.team_a_teammates = [[id for id in self.team_a_agents if id != current_agent]
                              for current_agent in self.team_a_agents]
        self.team_b_agents = [aid for aid in range(num_agents_team_a, num_agents_team_a + num_agents_team_b)]
        self.team_b_teammates = [[id for id in self.team_b_agents if id != current_agent]
                            for current_agent in self.team_b_agents]
        self.team_a_objectives_producer = ObjectivesProducer(agents_ids=self.team_a_agents,
                                                             num_valid_agents=max_num_valid_agents,
                                                             teammate_ids=self.team_a_teammates, **self.rules)
        self.team_b_objectives_producer = ObjectivesProducer(agents_ids=self.team_b_agents,
                                                             num_valid_agents=max_num_valid_agents,
                                                             teammate_ids=self.team_b_teammates, **self.rules)
        self.starter_objective = starter_objective
        self._finalizer = None
        self._mailbox: Optional[Mailbox] = None
        self._scheduler: Optional[SpotlightScheduler] = None
        self._st: Optional[ShortTermPlanner] = None
        self._lt: Optional[LongTermPlanner] = None

    # --- lifecycle hooks for DI environment ---
    def on_agent_init(self):
        if self._mailbox is not None:
            return
        self._mailbox = Mailbox()
        self._mailbox.command.set((0, 0))
        team_a_valid_objectives = self.team_a_objectives_producer.produce_valid_objectives()
        team_b_valid_objectives = self.team_b_objectives_producer.produce_valid_objectives()

        if self.team == 'A':
            team_objectives = team_a_valid_objectives
        else:
            team_objectives = team_b_valid_objectives

        own_objectives = [team_objectives[(self.agent_id, enum)] for enum in OBJECTIVE_ENUMS]
        self._st = ShortTermPlanner(mailbox=self._mailbox, objectives=own_objectives,
                                    starter_objective=self.starter_objective)

        team_a_valid_policies = {id_enum: ShortAgentPolicy(agent_id=id_enum[0],
                                                           objective_insert=objective)
                                 for id_enum, objective in team_a_valid_objectives.items()}

        team_a_policies_chooser = ShortPolicyChooser(agent_ids=self.team_a_agents[:self.max_num_valid_agents],
                                                     is_team_a=True,
                                                     width=self.width,
                                                     offensive_factors=OFFENSIVE_FACTORS,
                                                     short_term_policies=team_a_valid_policies)

        team_b_valid_policies = {id_enum: ShortAgentPolicy(agent_id=id_enum[0],
                                                           objective_insert=objective)
                                 for id_enum, objective in team_b_valid_objectives.items()}

        team_b_policies_chooser = ShortPolicyChooser(agent_ids=self.team_b_agents[:self.max_num_valid_agents],
                                                     is_team_a=False,
                                                     width=self.width,
                                                     offensive_factors=OFFENSIVE_FACTORS,
                                                     short_term_policies=team_b_valid_policies)


        team_a_pass_policies = {aid: ShortAgentPolicy(agent_id=aid, objective_insert=objective)
                         for aid, objective in self.team_a_objectives_producer.produce_pass_objectives().items()}

        team_b_pass_policies = {aid: ShortAgentPolicy(agent_id=aid, objective_insert=objective)
                         for aid, objective in self.team_b_objectives_producer.produce_pass_objectives().items()}

        self._lt = LongTermPlanner(agent_id=self.agent_id,
                                   team=self.team,
                                   num_agents_team_a=self.num_agents_team_a,
                                   num_agents_team_b=self.num_agents_team_b,
                                   mailbox=self._mailbox,
                                   team_a_policies_chooser=team_a_policies_chooser,
                                   team_b_policies_chooser=team_b_policies_chooser,
                                   team_a_pass_policies=team_a_pass_policies,
                                   team_b_pass_policies=team_b_pass_policies,
                                   rules=self.rules)
        self._scheduler = SpotlightScheduler(self._st, self._lt)

        self._scheduler.start()
        self._register_finalizers()

    def on_agent_close(self):
        self._signal_shutdown()
        self._thread_shutdown()
        self._join_briefly()

    def _signal_shutdown(self):
        if self._mailbox and not self._mailbox.shutdown.is_set():
            try:
                self._mailbox.shutdown.set()
            except Exception:
                pass

    def _thread_shutdown(self):
        if self._scheduler:
            try:
                self._scheduler.shutdown()
            except Exception:
                pass

    def _join_briefly(self, timeout: float = 0.2):
        try:
            if self._scheduler is not None:
                self._scheduler.join(timeout=timeout)
        except Exception:
            pass

    def _register_finalizers(self):
        atexit.register(self._thread_shutdown)  # stop scheduler
        atexit.register(self._signal_shutdown)  # set mailbox flag
        self._finalizer = weakref.finalize(self, type(self)._finalize_static, weakref.ref(self))

    @staticmethod
    def _finalize_static(self_ref):
        self = self_ref()
        if not self:
            return
        try:
            self._thread_shutdown()
            self._signal_shutdown()
        except Exception:
            pass

    # --- Immediate layer: called at 30–60 Hz ---
    def update(self, world_state: Dict) -> Command:
        if self._mailbox is None:
            self.on_agent_init()

        if not self._scheduler or not self._scheduler.is_alive():
            return 0, 0

        # publish latest frame (peeked by ST/LT)
        self._mailbox.latest_world_state.set(world_state)
        self._scheduler.trigger_st_cycle()
        # pop next command if available, otherwise last cmd
        return self._mailbox.command.get()

    # --- make pickling safe (threads are not pickled) ---
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_scheduler"] = None
        d["_mailbox"] = None
        d["_st"] = None
        d["_lt"] = None
        d["_finalizer"] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
