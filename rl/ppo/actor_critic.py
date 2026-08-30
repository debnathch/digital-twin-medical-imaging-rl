import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """Actor-Critic network for discrete action spaces.
    
    Separate actor and critic networks to prevent gradient interference.
    Uses orthogonal initialization for stable training.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Actor: obs -> logits
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        # Critic: obs -> value
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for stable PPO training."""
        for module in [self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)
        # Actor output layer: small gain for exploration
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        # Critic output layer: unit gain
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs)
    
    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action, return (action, log_prob, value, entropy)."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1), dist.entropy()
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions, return (log_prob, value, entropy)."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), value.squeeze(-1), dist.entropy()
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)
    
    def get_action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action probabilities for explainability."""
        logits = self.actor(obs)
        return torch.softmax(logits, dim=-1)
