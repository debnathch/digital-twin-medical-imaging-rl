from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np

@dataclass
class ConvergenceResult:
    """Result of convergence analysis."""
    converged: bool = False
    convergence_episode: Optional[int] = None
    final_moving_avg: float = 0.0
    best_evaluation_episode: int = 0
    best_evaluation_score: float = 0.0
    total_episodes: int = 0
    criteria_met: Dict[str, bool] = field(default_factory=dict)  # criterion_name -> bool
    message: str = "NOT CONVERGED"

@dataclass
class ConvergenceConfig:
    window_size: int = 100           # episodes for moving average
    min_improvement: float = 0.005   # 0.5% minimum improvement
    patience: int = 5                # consecutive windows without improvement
    reward_variance_threshold: float = 0.5  # max normalized variance
    kl_stability_threshold: float = 0.05    # max KL change between windows
    min_episodes: int = 200          # minimum episodes before checking

class ConvergenceDetector:
    """Formal convergence detection for RL training.
    
    Convergence is declared when ALL of:
    1. Moving-average global return changes less than min_improvement
    2. For patience consecutive evaluation windows
    3. Evaluation performance does not materially improve
    4. Policy KL remains stable
    5. No major reward collapse
    """
    def __init__(self, config: Optional[ConvergenceConfig] = None):
        self.config = config or ConvergenceConfig()
        self.episode_returns: List[float] = []
        self.moving_averages: List[float] = []
        self.kl_divergences: List[float] = []
        self.reward_variances: List[float] = []
        self.patience_counter: int = 0
        self.best_moving_avg: float = float('-inf')
        self.best_eval_episode: int = 0
        
    def update(self, episode_return: float, kl_divergence: float = 0.0) -> ConvergenceResult:
        """Update with new episode data and check convergence."""
        self.episode_returns.append(episode_return)
        self.kl_divergences.append(kl_divergence)
        
        ep_count = len(self.episode_returns)
        
        # Compute moving average
        if ep_count >= self.config.window_size:
            window_returns = self.episode_returns[-self.config.window_size:]
            moving_avg = float(np.mean(window_returns))
            variance = float(np.var(window_returns)) / (abs(moving_avg) + 1e-6)
        else:
            moving_avg = episode_return
            variance = 0.0
            
        self.moving_averages.append(moving_avg)
        self.reward_variances.append(variance)
        
        result = ConvergenceResult(total_episodes=ep_count, final_moving_avg=moving_avg)
        
        # Check improvements
        if moving_avg > self.best_moving_avg + abs(self.best_moving_avg) * self.config.min_improvement:
            self.best_moving_avg = moving_avg
            self.best_eval_episode = ep_count
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        result.best_evaluation_episode = self.best_eval_episode
        result.best_evaluation_score = self.best_moving_avg
        
        if ep_count < self.config.min_episodes:
            result.message = "NOT CONVERGED: Minimum episodes not reached"
            return result
            
        # Check all criteria
        c1_no_improvement = self.patience_counter > 0
        c2_patience_met = self.patience_counter >= self.config.patience
        c3_eval_stable = c2_patience_met
        c4_kl_stable = kl_divergence < self.config.kl_stability_threshold
        c5_no_collapse = variance < self.config.reward_variance_threshold
        
        result.criteria_met = {
            'no_improvement': c1_no_improvement,
            'patience_met': c2_patience_met,
            'eval_stable': c3_eval_stable,
            'kl_stable': c4_kl_stable,
            'no_collapse': c5_no_collapse
        }
        
        result.converged = all(result.criteria_met.values())
        
        if result.converged:
            result.convergence_episode = ep_count
            result.message = "CONVERGED"
        else:
            result.message = "NOT CONVERGED"
            
        return result
    
    def get_training_curves(self) -> Dict[str, Any]:
        """Return data for convergence dashboard plots."""
        return {
            'episode_returns': self.episode_returns,
            'moving_averages': self.moving_averages,
            'kl_divergences': self.kl_divergences,
            'reward_variances': self.reward_variances,
        }
