"""AI Model Agent — Level 2 agent.

Selects the appropriate AI model for inference.
"""

import numpy as np
from agents.observations import AIModelObservationBuilder
from simulator.state.global_state import GlobalState


class AIModelAgent:
    """Level 2 AI Model Selection Agent.

    Action space: index into available AI models (0 .. MAX_MODELS-1).
    """
    MAX_MODELS = 5
    NUM_ACTIONS = 5

    def __init__(self):
        self.obs_builder = AIModelObservationBuilder()

    def get_observation(self, global_state: GlobalState,
                        workflow_context: np.ndarray = None) -> np.ndarray:
        return self.obs_builder.build(global_state, workflow_context=workflow_context)
