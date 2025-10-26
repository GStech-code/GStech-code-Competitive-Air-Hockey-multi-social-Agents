import threading
from .short_term_planner import ShortTermPlanner
from .long_term_planner import LongTermPlanner

class SpotlightMode:
    IDLE = 0   # nothing to do
    ST = 1   # short-term spotlight
    LT = 2   # long-term spotlight

class SpotlightScheduler(threading.Thread):
    """
    Cooperative scheduler for ShortTerm and LongTerm planners.
    - Immediate work (policy.update) is synchronous and outside this thread.
    - Spotlight toggles between ST and LT.
    - Emergency spotlight lets ST preempt everything until it clears.
    """
    def __init__(self, st: ShortTermPlanner, lt: LongTermPlanner):
        super().__init__(daemon=True, name="Scheduler")
        self.st = st
        self.lt = lt

        self._mode = SpotlightMode.IDLE
        self.emergency_status = False
        self._cv = threading.Condition()
        self._shutdown = False

    def trigger_st_cycle(self):
        """Called by Policy.update(): start a short-term spotlight window."""
        with self._cv:
            self._mode = SpotlightMode.ST
            self._cv.notify()

    def shutdown(self):
        with self._cv:
            self._shutdown = True
            self._cv.notify_all()

    def run(self):
        while True:
            with self._cv:
                while not self._shutdown and self._mode == SpotlightMode.IDLE:
                    self._cv.wait()
                if self._shutdown:
                    return
                mode = self._mode

            if mode == SpotlightMode.ST:
                self.st.step()
                self._mode = SpotlightMode.LT
                self.lt.new_ws_step()

            elif mode == SpotlightMode.LT:
                self.lt.step()
