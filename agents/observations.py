"""Agent observation builders — heterogeneous observations for each agent.

Each agent receives a DIFFERENT observation from the global state.
Workflow Agent (Level 1) gets the broadest view.
Level 2 agents get focused observations plus workflow context.

SYNTHETIC / RESEARCH DATA ONLY
"""

import numpy as np
from simulator.state.global_state import GlobalState


class WorkflowObservationBuilder:
    """Observation for Workflow Agent (Level 1).

    O_W = [QueueState(8), ResourceState(4), ModelState(5), SLAState(3),
           WorkflowContext(10)] = 30 dims
    """
    OBS_DIM = 30

    def build(self, global_state: GlobalState, **kwargs) -> np.ndarray:
        parts = [
            global_state.queue.to_array(),          # 8
            global_state.human_resource.to_array(),  # 4
            global_state.model.to_array(),           # 5
            global_state.sla.to_array(),             # 3
            global_state.workflow.to_array(),         # 10
        ]
        obs = np.concatenate(parts).astype(np.float32)
        # Pad or trim to OBS_DIM
        if len(obs) < self.OBS_DIM:
            obs = np.pad(obs, (0, self.OBS_DIM - len(obs)))
        return obs[:self.OBS_DIM]


class QueueObservationBuilder:
    """Observation for Queue Agent (Level 2).

    O_Q = [QueueState(8), SLAState(3), WorkflowContext(5-one-hot)] = 16 dims
    Focused on queue dynamics.
    """
    OBS_DIM = 16

    def build(self, global_state: GlobalState, workflow_context: np.ndarray = None, **kwargs) -> np.ndarray:
        parts = [
            global_state.queue.to_array(),  # 8
            global_state.sla.to_array(),    # 3
        ]
        obs = np.concatenate(parts).astype(np.float32)
        # Append workflow context (one-hot of workflow strategy)
        if workflow_context is not None:
            obs = np.concatenate([obs, workflow_context[:5]])
        else:
            obs = np.concatenate([obs, np.zeros(5, dtype=np.float32)])
        if len(obs) < self.OBS_DIM:
            obs = np.pad(obs, (0, self.OBS_DIM - len(obs)))
        return obs[:self.OBS_DIM]


class ResourceObservationBuilder:
    """Observation for Resource Agent (Level 2).

    O_R = [HumanResource(4), ComputeState(4), WorkflowContext(5)] = 13 dims
    Focused on scanner/radiologist/GPU availability.
    """
    OBS_DIM = 13

    def build(self, global_state: GlobalState, workflow_context: np.ndarray = None, **kwargs) -> np.ndarray:
        parts = [
            global_state.human_resource.to_array(),  # 4
            global_state.compute.to_array(),          # 4
        ]
        obs = np.concatenate(parts).astype(np.float32)
        if workflow_context is not None:
            obs = np.concatenate([obs, workflow_context[:5]])
        else:
            obs = np.concatenate([obs, np.zeros(5, dtype=np.float32)])
        if len(obs) < self.OBS_DIM:
            obs = np.pad(obs, (0, self.OBS_DIM - len(obs)))
        return obs[:self.OBS_DIM]


class AIModelObservationBuilder:
    """Observation for AI Model Agent (Level 2).

    O_M = [ModelState(5), ComputeState(4), WorkflowContext(5)] = 14 dims
    Focused on model availability, suitability, GPU/memory.
    """
    OBS_DIM = 14

    def build(self, global_state: GlobalState, workflow_context: np.ndarray = None, **kwargs) -> np.ndarray:
        parts = [
            global_state.model.to_array(),   # 5
            global_state.compute.to_array(),  # 4
        ]
        obs = np.concatenate(parts).astype(np.float32)
        if workflow_context is not None:
            obs = np.concatenate([obs, workflow_context[:5]])
        else:
            obs = np.concatenate([obs, np.zeros(5, dtype=np.float32)])
        if len(obs) < self.OBS_DIM:
            obs = np.pad(obs, (0, self.OBS_DIM - len(obs)))
        return obs[:self.OBS_DIM]
