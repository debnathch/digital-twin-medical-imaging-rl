"""Queue Agent — Level 2 agent.

Selects the next study to process from the waiting queue.
"""

import numpy as np
from agents.observations import QueueObservationBuilder
from simulator.state.global_state import GlobalState


class QueueAgent:
    """Level 2 Queue Agent.

    Action space: index into top-K candidate studies (0 .. MAX_CANDIDATES-1).
    """
    MAX_CANDIDATES = 8
    NUM_ACTIONS = 8

    def __init__(self):
        self.obs_builder = QueueObservationBuilder()

    def get_observation(self, global_state: GlobalState,
                        workflow_context: np.ndarray = None) -> np.ndarray:
        return self.obs_builder.build(global_state, workflow_context=workflow_context)
