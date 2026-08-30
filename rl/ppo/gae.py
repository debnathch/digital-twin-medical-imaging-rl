import torch

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: float = 0.0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.
    
    A_hat_t = sum_l (gamma * lambda)^l * delta_{t+l}
    where delta_t = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
    
    Returns:
        advantages: GAE advantages
        returns: advantages + values (targets for value function)
    """
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_val = values[t + 1]
            
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
        
    returns = advantages + values
    return advantages, returns
