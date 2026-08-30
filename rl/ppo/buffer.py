import torch
import numpy as np
from dataclasses import dataclass, field

@dataclass
class RolloutBuffer:
    """Stores rollout data for a single agent during an episode."""
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    
    def add(self, obs, action, log_prob, reward, value, done):
        """Add a single transition."""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def get(self) -> dict[str, torch.Tensor]:
        """Convert to tensors for training. Returns dict with all fields as tensors."""
        # Convert lists to numpy arrays first for efficiency, then to tensors
        data = {
            'observations': torch.FloatTensor(np.array(self.observations)),
            'actions': torch.FloatTensor(np.array(self.actions)),
            'log_probs': torch.FloatTensor(np.array(self.log_probs)),
            'rewards': torch.FloatTensor(np.array(self.rewards)),
            'values': torch.FloatTensor(np.array(self.values)),
            'dones': torch.FloatTensor(np.array(self.dones)),
        }
        return data
    
    def clear(self):
        """Clear all stored data."""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self) -> int:
        return len(self.rewards)


class MultiAgentRolloutBuffer:
    """Manages rollout buffers for multiple heterogeneous agents."""
    def __init__(self, agent_names: list[str]):
        self.buffers: dict[str, RolloutBuffer] = {name: RolloutBuffer() for name in agent_names}
    
    def add(self, agent_name: str, obs, action, log_prob, reward, value, done):
        self.buffers[agent_name].add(obs, action, log_prob, reward, value, done)

    def get(self, agent_name: str) -> dict[str, torch.Tensor]:
        return self.buffers[agent_name].get()

    def clear_all(self):
        for buffer in self.buffers.values():
            buffer.clear()

    def __len__(self) -> int:
        if not self.buffers:
            return 0
        return len(next(iter(self.buffers.values())))
