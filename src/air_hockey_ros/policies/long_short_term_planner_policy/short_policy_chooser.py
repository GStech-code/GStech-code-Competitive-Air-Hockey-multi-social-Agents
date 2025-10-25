# short_policy_chooser.py
from typing import List, Dict, Tuple
from .short_term_objectives import ObjectiveEnum
from .short_agent_policy import ShortAgentPolicy

class ShortPolicyChooser:
    def __init__(self, agent_ids: List[int],
                 width: float,
                 offensive_factors: List[float],
                 short_term_policies: Dict[Tuple[int, ObjectiveEnum], ShortAgentPolicy]):
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        self.inv_w = 1 / width
        self.offensive_factors = {
            agent_id: factor
            for agent_id, factor in zip(agent_ids, offensive_factors[:self.num_agents])
        }
        self._short_term_policies = short_term_policies  # keep field private-ish

    def get_policies(self) -> Dict[Tuple[int, ObjectiveEnum], ShortAgentPolicy]:
        return self._short_term_policies

    def choose_objectives(self, agents_x: List[float], combinations: List[ObjectiveEnum]) -> Dict[int, ObjectiveEnum]:
        """
        Pure mapping: decide which agent gets which objective (enum).
        This is the canonical 'chooser' responsibility.
        """
        ordering = sorted(
            self.agent_ids,
            key=lambda aid: (agents_x[aid] * self.inv_w * self.offensive_factors[aid], aid),
        )
        return {aid: enum for aid, enum in zip(ordering, combinations)}

    def get_assignments(self, agents_x: List[float], combinations: List[ObjectiveEnum]) -> Dict[int, ShortAgentPolicy]:
        """
        Backwards-compatible helper that returns policies, built from choose_objectives.
        """
        mapping = self.choose_objectives(agents_x, combinations)
        return {aid: self._short_term_policies[(aid, enum)] for aid, enum in mapping.items()}
