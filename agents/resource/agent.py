"""Resource Agent — Level 2 agent.

Assigns scanner and radiologist to the selected study.
"""

import numpy as np
from agents.observations import ResourceObservationBuilder
from simulator.state.global_state import GlobalState


class ResourceAgent:
    """Level 2 Resource Agent.

    Action space: combined scanner + radiologist assignment.
    action = scanner_index * MAX_RADIOLOGISTS + radiologist_index
    """
    MAX_SCANNERS = 6
    MAX_RADIOLOGISTS = 10
    NUM_ACTIONS = 60  # 6 * 10

    def __init__(self):
        self.obs_builder = ResourceObservationBuilder()

    def get_observation(self, global_state: GlobalState,
                        workflow_context: np.ndarray = None) -> np.ndarray:
        return self.obs_builder.build(global_state, workflow_context=workflow_context)

    def decode_action(self, action: int) -> tuple:
        """Decode combined action into (scanner_index, radiologist_index)."""
        action = action % self.NUM_ACTIONS
        scanner_idx = action // self.MAX_RADIOLOGISTS
        rad_idx = action % self.MAX_RADIOLOGISTS
        return scanner_idx, rad_idx
