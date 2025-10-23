from typing import Optional, Dict, Tuple, List
from bus import Mailbox
from short_term_objectives import Objective, OBJECTIVES_DICT, ObjectiveEnum

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
        self.latest_world_state = mailbox.latest_world_state
        self.command = mailbox.command

        self.current_objective_enum = starting_objective_enum
        self.rules = params.get('rules', {})
        self.objectives: List[Objective] = [objective_class(agent_id, teammate_ids,
                                                            self.rules, **params.get(objective_enum, {}))
                                            for objective_enum, objective_class in OBJECTIVES_DICT.items()]


    def step(self) -> None:
        if self.status_change_flag.acknowledge():
            self.current_objective_enum = self.latest_instruction.get()
        ws = self.latest_world_state.get()
        cmd = self.objectives[self.current_objective_enum].step(ws)
        self.command.set(cmd)

