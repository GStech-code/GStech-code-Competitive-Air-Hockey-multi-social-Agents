from typing import Optional, Dict, Tuple

class Objective:
    def __init__(self, agent_id, teammate_ids, commands, rules: Dict, **params):
        self.agent_id = agent_id
        self.teammate_ids = teammate_ids
        self.commands = commands
        self.last_ws: Dict = None
        self.rules = rules

    def intro_step(self, ws: Dict, **params):
        self.commands.flush_limit(2)
        #frame = self.latest_frame.get()

    def new_ws_step(self, ws: Dict):
        self.last_ws = ws

    def continue_step(self):
        if self.commands.is_full():
            return
        #frame = self.latest_frame.get()

    def emergency_step(self, ws: Dict, **params):
        #frame = self.latest_frame.get()
        self.commands.clear()
        self.commands.push((0, 0))
