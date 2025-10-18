from typing import Optional, Dict, Tuple, Callable

def noops():
    pass
class Objective:
    def __init__(self, agent_id, teammate_ids, commands, rules: Dict, **params):
        self.agent_id = agent_id
        self.teammate_ids = teammate_ids
        self.commands = commands
        self.last_ws: Dict = None
        self.long_term_mode: Callable = noops
        self.rules = rules


    def set_long_term_mode_func(self, func: Callable):
        self.long_term_mode = func
    def intro_step(self, ws: Dict):
        self.commands.flush_limit(2)
        #frame = self.latest_frame.get()

    def new_ws_step(self, ws: Dict):
        self.last_ws = ws

    def continue_step(self):
        if self.commands.is_full():
            return
        #frame = self.latest_frame.get()

    def emergency_step(self, ws: Dict):
        #frame = self.latest_frame.get()
        self.commands.clear()
        self.commands.push((0, 0))
