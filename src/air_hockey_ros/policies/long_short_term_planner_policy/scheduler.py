import threading
import time

now_ns = time.monotonic_ns

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
    def __init__(self, st, lt, st_budget_ms=2):
        super().__init__(daemon=True, name="Scheduler")
        self.st = st
        self.lt = lt
        self.st_budget_ns = int(st_budget_ms * 1e6)

        self._mode = SpotlightMode.LT
        self._deadline_ns = 0
        self.emergency_status = False
        self._cv = threading.Condition()
        self._shutdown = False

    def trigger_st_cycle(self):
        """Called by Policy.update(): start a short-term spotlight window."""
        with self._cv:
            self._mode = SpotlightMode.ST
            self._deadline_ns = now_ns() + self.st_budget_ns
            self._cv.notify()

    def trigger_emergency(self):
        """Called by ST planner: emergency spotlight until it clears itself."""
        self.emergency_status = True
        with self._cv:
            self._cv.notify()

    def get_emergency_func(self):
        return self.trigger_emergency

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
                deadline = self._deadline_ns

            if self.emergency_status:
                self.st.emergency_step()
                with self._cv:
                    self._mode = SpotlightMode.LT
                self.emergency_status = False

            elif mode == SpotlightMode.ST:
                self.st.step()
                if now_ns() >= deadline:
                    with self._cv:
                        self._mode = SpotlightMode.LT

            elif mode == SpotlightMode.LT:
                self.lt.step()
