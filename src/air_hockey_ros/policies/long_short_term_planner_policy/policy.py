from typing import Dict, Tuple, Optional
import time
import atexit
import weakref
from .bus import Mailbox
from .scheduler import SpotlightScheduler
from .short_term_planner import ShortTermPlanner
from .long_term_planner import LongTermPlanner
from air_hockey_ros import TeamPolicy, register_policy, AgentPolicy

Command = Tuple[int, int]

now_ns = time.monotonic_ns

@register_policy("long_short_term_planner")
class Policy3Layer(TeamPolicy):
    """Pickle‑friendly: threads are created/started in on_agent_init()."""

    def __init__(self, st_min_period_ms: int = 2, cmd_capacity: int = 32):
        self.st_min_period_ms=st_min_period_ms
        self.cmd_capacity=cmd_capacity
        self._finalizer = None
        self._mailbox: Optional[Mailbox] = None
        self._scheduler: Optional[SpotlightScheduler] = None
        self._st: Optional[ShortTermPlanner] = None
        self._lt: Optional[LongTermPlanner] = None
        self._last_cmd: Command = (0, 0)

    # --- lifecycle hooks for DI environment ---
    def on_agent_init(self):
        if self._mailbox is not None:
            return
        self._mailbox = Mailbox(cmd_capacity=self.cmd_capacity)
        self._st = ShortTermPlanner(self._mailbox)
        self._lt = LongTermPlanner(self._mailbox)
        self._scheduler = SpotlightScheduler(self._st, self._lt, st_budget_ms=self.st_min_period_ms)
        emergency_func = self._scheduler.get_emergency_func()
        self._st.set_emergency_func(emergency_func)
        self._lt.set_emergency_func(emergency_func)

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
        if not self: return
        try:
            self._thread_shutdown()
            self._signal_shutdown()
        except Exception:
            pass

    # --- Immediate layer: called at 30–60 Hz ---
    def update(self, world_state: Dict) -> Tuple[int, int]:
        if self._mailbox is None:
            self.on_agent_init()

        if not self._scheduler or not self._scheduler.is_alive():
            return self._last_cmd

        # publish latest frame (peeked by ST/LT)
        self._mailbox.latest_world_state.set(world_state)
        if self._scheduler.emergency_status:
            return 0, 0  # TODO: Decide if to send (0, 0) or last_cmd or commands.pop
        self._scheduler.trigger_st_cycle()

        # pop next command if available, otherwise (0,0)
        cmd = self._mailbox.commands.pop() or (0, 0)
        self._last_cmd = cmd
        return cmd

    def last_cmd(self) -> Command:
        return self._last_cmd

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
