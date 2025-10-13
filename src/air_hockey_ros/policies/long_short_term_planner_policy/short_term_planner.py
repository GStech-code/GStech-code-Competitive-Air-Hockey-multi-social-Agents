from typing import Optional, Dict, Tuple
from bus import Mailbox
from short_term_objectives import Objective, OBJECTIVES, ObjectiveEnum

class ShortTermPlanner:
    """
    Stateless-ish ST that advances in tiny steps.
    Called by SpotlightScheduler:
      - emergency_step  when in EMERGENCY spotlight
      - step()                during ST spotlight window
    """
    def __init__(self, agent_id, teammate_ids, mailbox: Mailbox,
                 starting_objective_enum: ObjectiveEnum = ObjectiveEnum.DEFEND_LINE,
                 **params):
        self.latest_instruction = mailbox.latest_instruction
        self.status_change_flag = mailbox.status_change_flag
        self.latest_frame = mailbox.latest_frame
        self.commands = mailbox.commands

        self.current_objective_enum = starting_objective_enum
        self.long_term_mode_func = None
        self.instruction: Optional[Tuple[ObjectiveEnum, Dict]] = None
        self.commands_max_target = params.get('commands_target', self.commands.get_capacity())
        self.commands_min_target = params.get('commands_min_target', self.commands.get_capacity() // 2)
        self.commands_threshold = params.get('commands_threshold', 2)
        self.last_seq_id = -1
        self.rules = params.get('rules', {})
        self.objectives = [objective(agent_id, teammate_ids, mailbox.latest_frame, mailbox.commands,
                                     self.rules, **params.get(objective_enum, {}))
                           for objective, objective_enum in zip(OBJECTIVES, ObjectiveEnum)]

    def set_long_term_mode_func(self, long_term_mode_func):
        self.long_term_mode_func = long_term_mode_func
        for objective in self.objectives:
            objective.set_long_term_mode_func(long_term_mode_func)

    def step(self) -> None:
        if self.status_change_flag.acknowledge():
            self.current_objective_enum, params = self.latest_instruction.get()
            seq_id, ws = self.latest_frame.get()
            self.commands.flush_limit()
            self.objectives[self.current_objective_enum].intro_step(ws, **params)
            self.last_seq_id = seq_id
        else:
            existing_commands = self.commands.get_size()
            if existing_commands >= self.commands_max_target:
                self.long_term_mode_func()
                return
            seq_id, ws = self.latest_frame.get()
            if self.last_seq_id > seq_id:
                self.objectives[self.current_objective_enum].new_ws_step(ws)
                self.last_seq_id = seq_id
                return
            if existing_commands >= self.commands_min_target:
                self.long_term_mode_func()
                return
            self.objectives[self.current_objective_enum].continue_step()

    def emergency_step(self) -> None:
        self.current_objective_enum, params = self.latest_instruction.get()
        seq_id, ws = self.latest_frame.get()
        self.commands.clear()
        self.objectives[self.current_objective_enum].emergency_step(ws, **params)
        self.last_seq_id = seq_id
