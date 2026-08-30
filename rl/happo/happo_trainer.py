import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

from rl.ppo.actor_critic import ActorCritic
from rl.ppo.buffer import RolloutBuffer, MultiAgentRolloutBuffer
from rl.ppo.gae import compute_gae

@dataclass
class AgentConfig:
    name: str
    obs_dim: int
    act_dim: int
    hidden_dim: int = 64
    learning_rate: float = 3e-4

@dataclass
class HAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20
    ppo_epochs: int = 10
    batch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Agent-specific configs
    agent_configs: dict[str, AgentConfig] = field(default_factory=dict)

@dataclass
class TrainingStats:
    """Per-agent training statistics for one update."""
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0

class HAPPOTrainer:
    """Heterogeneous-Agent Proximal Policy Optimization.
    
    Implements HAPPO sequential update mechanism:
    1. Update agents in hierarchical order: workflow -> queue -> resource -> model
    2. Each agent's PPO update uses compound importance ratio M
    3. M is updated after each agent's policy is updated
    
    This is the MULTI-AGENT POLICY OPTIMIZATION MECHANISM, not the architecture.
    The hierarchy (Workflow=L1, Queue/Resource/Model=L2) is the APPLICATION ARCHITECTURE.
    HAPPO is the optimization method that respects agent heterogeneity.
    """
    
    AGENT_ORDER = ['workflow', 'queue', 'resource', 'model']
    
    def __init__(self, config: HAPPOConfig):
        self.config = config
        self.policies: dict[str, ActorCritic] = {}
        self.optimizers: dict[str, torch.optim.Adam] = {}
        
        for name in self.AGENT_ORDER:
            agent_cfg = config.agent_configs[name]
            policy = ActorCritic(agent_cfg.obs_dim, agent_cfg.act_dim, agent_cfg.hidden_dim)
            self.policies[name] = policy
            self.optimizers[name] = torch.optim.Adam(policy.parameters(), lr=agent_cfg.learning_rate)
    
    def sequential_update(self, buffers: MultiAgentRolloutBuffer) -> dict[str, TrainingStats]:
        """HAPPO sequential update across all agents.
        
        Key insight from HAPPO paper:
        - Agents are updated sequentially in a fixed order
        - A compound importance ratio M tracks the cumulative effect of earlier agent updates
        - Each agent's clipped surrogate is scaled by M
        - After updating agent i, M is updated: M = M * (new_prob_i / old_prob_i)
        
        This provides monotonic improvement guarantees for the joint policy.
        """
        all_stats = {}
        
        # Get batch data for all agents
        agent_data = {}
        for name in self.AGENT_ORDER:
            data = buffers.get(name)
            if len(data['rewards']) == 0:
                continue
            # Compute GAE
            advantages, returns = compute_gae(
                data['rewards'], data['values'], data['dones'],
                gamma=self.config.gamma, gae_lambda=self.config.gae_lambda
            )
            agent_data[name] = {
                'obs': data['observations'],
                'actions': data['actions'],
                'old_log_probs': data['log_probs'],
                'advantages': advantages,
                'returns': returns,
            }
        
        if not agent_data:
            return all_stats
            
        batch_size = min(len(next(iter(agent_data.values()))['obs']), self.config.batch_size)
        
        # Initialize compound importance ratio M = 1 for all transitions
        num_transitions = len(next(iter(agent_data.values()))['obs'])
        M = torch.ones(num_transitions)
        
        # Sequential update in hierarchical order
        for name in self.AGENT_ORDER:
            if name not in agent_data:
                continue
            
            data = agent_data[name]
            policy = self.policies[name]
            optimizer = self.optimizers[name]
            
            # PPO update with HAPPO compound ratio
            stats = self._ppo_update_with_happo(
                policy, optimizer, data, M,
                epochs=self.config.ppo_epochs,
                batch_size=batch_size,
                clip_epsilon=self.config.clip_epsilon,
                entropy_coef=self.config.entropy_coef,
                value_coef=self.config.value_coef,
                max_grad_norm=self.config.max_grad_norm,
            )
            all_stats[name] = stats
            
            # Update compound ratio M for next agent
            with torch.no_grad():
                new_log_probs, _, _ = policy.evaluate_actions(
                    data['obs'], data['actions']
                )
                M = M * torch.exp(new_log_probs - data['old_log_probs'])
                M = M.detach()
        
        return all_stats
    
    def _ppo_update_with_happo(self, policy, optimizer, data, M, 
                                epochs, batch_size, clip_epsilon,
                                entropy_coef, value_coef, max_grad_norm) -> TrainingStats:
        """Standard PPO clipped surrogate update, scaled by HAPPO compound ratio M."""
        obs = data['obs']
        actions = data['actions']
        old_log_probs = data['old_log_probs']
        advantages = data['advantages']
        returns = data['returns']
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        num_updates = 0
        
        for _ in range(epochs):
            indices = np.random.permutation(len(obs))
            for start in range(0, len(obs), batch_size):
                end = min(start + batch_size, len(obs))
                idx = indices[start:end]
                
                new_log_probs, values, entropy = policy.evaluate_actions(
                    obs[idx], actions[idx]
                )
                
                # HAPPO: scale advantages by compound ratio M
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                scaled_adv = advantages[idx] * M[idx].detach()
                
                surr1 = ratio * scaled_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * scaled_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * ((values - returns[idx]) ** 2).mean()
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += (old_log_probs[idx] - new_log_probs).mean().item()
                total_clip_frac += ((ratio - 1).abs() > clip_epsilon).float().mean().item()
                num_updates += 1
        
        return TrainingStats(
            policy_loss=total_policy_loss / max(num_updates, 1),
            value_loss=total_value_loss / max(num_updates, 1),
            entropy=total_entropy / max(num_updates, 1),
            approx_kl=total_kl / max(num_updates, 1),
            clip_fraction=total_clip_frac / max(num_updates, 1),
        )
    
    def get_actions(self, observations: dict[str, torch.Tensor]) -> dict[str, dict]:
        """Get actions from all policies given observations.
        Returns dict of {agent_name: {action, log_prob, value, entropy, action_probs}}.
        """
        results = {}
        for name in self.AGENT_ORDER:
            if name in observations:
                obs = observations[name]
                if not isinstance(obs, torch.Tensor):
                    obs = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, value, entropy = self.policies[name].get_action(obs)
                    action_probs = self.policies[name].get_action_probs(obs)
                results[name] = {
                    'action': action.item(),
                    'log_prob': log_prob.item(),
                    'value': value.item(),
                    'entropy': entropy.item(),
                    'action_probs': action_probs.squeeze(0).numpy(),
                }
        return results
    
    def save_checkpoint(self, path: str):
        """Save all policies and optimizers."""
        checkpoint = {
            'config': self.config,
            'policies': {name: p.state_dict() for name, p in self.policies.items()},
            'optimizers': {name: o.state_dict() for name, o in self.optimizers.items()},
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load all policies and optimizers."""
        checkpoint = torch.load(path, map_location='cpu')
        for name in self.AGENT_ORDER:
            self.policies[name].load_state_dict(checkpoint['policies'][name])
            self.optimizers[name].load_state_dict(checkpoint['optimizers'][name])
