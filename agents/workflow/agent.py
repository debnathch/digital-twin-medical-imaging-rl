"""Workflow Agent — Level 1 agent.

Establishes high-level workflow strategy that guides Level 2 agents.
"""

import numpy as np
from agents.observations import WorkflowObservationBuilder
from simulator.state.global_state import GlobalState


class WorkflowAgent:
    """Level 1 Workflow Agent.

    Action space (5 discrete actions):
      0 = balanced_throughput
      1 = urgent_priority
      2 = sla_protection
      3 = resource_balancing
      4 = compute_aware
    """
    ACTION_NAMES = [
        "balanced_throughput", "urgent_priority", "sla_protection",
        "resource_balancing", "compute_aware",
    ]
    NUM_ACTIONS = 5

    def __init__(self):
        self.obs_builder = WorkflowObservationBuilder()
        self.current_strategy: int = 0

    def get_observation(self, global_state: GlobalState) -> np.ndarray:
        return self.obs_builder.build(global_state)

    def set_strategy(self, action: int):
        self.current_strategy = action % self.NUM_ACTIONS

    def get_strategy_context(self) -> np.ndarray:
        """One-hot encoded strategy for Level 2 agents."""
        ctx = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        ctx[self.current_strategy] = 1.0
        return ctx
