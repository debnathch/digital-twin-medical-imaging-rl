import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

@dataclass
class ComparisonResult:
    """Statistical comparison between two policies."""
    metric_name: str = ""
    policy_a_name: str = ""
    policy_b_name: str = ""
    mean_a: float = 0.0
    mean_b: float = 0.0
    std_a: float = 0.0
    std_b: float = 0.0
    mean_difference: float = 0.0
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    welch_t_statistic: float = 0.0
    welch_p_value: float = 1.0
    mann_whitney_p_value: float = 1.0
    cohens_d: float = 0.0
    hedges_g: float = 0.0
    significant_at_05: bool = False
    winner: str = ""  # "A", "B", or "No significant difference"
    
    def to_dict(self) -> Dict[str, Any]: 
        return {k: v for k, v in self.__dict__.items()}

class StatisticalAnalyzer:
    """Statistical comparison of RL policies.
    
    Implements:
    - Welch's t-test (unequal variances)
    - Mann-Whitney U test (non-parametric)
    - Bootstrap confidence intervals
    - Cohen's d and Hedges' g effect sizes
    - Paired comparisons across identical seeds
    """
    
    def __init__(self, n_bootstrap: int = 10000):
        self.n_bootstrap = n_bootstrap
    
    def compare(self, scores_a: np.ndarray, scores_b: np.ndarray,
                metric_name: str = "", policy_a_name: str = "A",
                policy_b_name: str = "B",
                higher_is_better: bool = True) -> ComparisonResult:
        """Full statistical comparison between two sets of scores."""
        result = ComparisonResult(
            metric_name=metric_name,
            policy_a_name=policy_a_name,
            policy_b_name=policy_b_name
        )
        
        if len(scores_a) == 0 or len(scores_b) == 0:
            return result
            
        result.mean_a = float(np.mean(scores_a))
        result.mean_b = float(np.mean(scores_b))
        result.std_a = float(np.std(scores_a, ddof=1) if len(scores_a) > 1 else 0)
        result.std_b = float(np.std(scores_b, ddof=1) if len(scores_b) > 1 else 0)
        result.mean_difference = result.mean_a - result.mean_b
        
        # 1. Welch's t-test
        if len(scores_a) > 1 and len(scores_b) > 1:
            t_stat, p_val = stats.ttest_ind(scores_a, scores_b, equal_var=False)
            result.welch_t_statistic = float(t_stat)
            result.welch_p_value = float(p_val)
            
            # 2. Mann-Whitney U
            try:
                u_stat, u_pval = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
                result.mann_whitney_p_value = float(u_pval)
            except ValueError:
                result.mann_whitney_p_value = 1.0
                
            # 4. Cohen's d
            pooled_std = np.sqrt(((len(scores_a)-1)*result.std_a**2 + (len(scores_b)-1)*result.std_b**2) / 
                                (len(scores_a) + len(scores_b) - 2))
            if pooled_std > 0:
                result.cohens_d = result.mean_difference / pooled_std
            
            # 5. Hedges' g
            n = len(scores_a) + len(scores_b)
            correction = 1 - (3 / (4 * n - 9))
            result.hedges_g = result.cohens_d * correction
            
        # 3. Bootstrap CI of mean difference
        diffs = scores_a - scores_b if len(scores_a) == len(scores_b) else None
        if diffs is not None and len(diffs) > 1:
            lower, upper = self.bootstrap_ci(diffs)
            result.ci_95_lower = lower
            result.ci_95_upper = upper
        elif len(scores_a) > 1 and len(scores_b) > 1:
            # Independent bootstrap
            mean_diffs = []
            for _ in range(min(self.n_bootstrap, 1000)): # smaller n for independent to save time
                sa = np.random.choice(scores_a, size=len(scores_a), replace=True)
                sb = np.random.choice(scores_b, size=len(scores_b), replace=True)
                mean_diffs.append(np.mean(sa) - np.mean(sb))
            result.ci_95_lower, result.ci_95_upper = np.percentile(mean_diffs, [2.5, 97.5])
            
        # 6. Determine winner
        result.significant_at_05 = result.welch_p_value < 0.05
        
        if result.significant_at_05:
            if higher_is_better:
                result.winner = policy_a_name if result.mean_a > result.mean_b else policy_b_name
            else:
                result.winner = policy_a_name if result.mean_a < result.mean_b else policy_b_name
        else:
            result.winner = "No significant difference"
            
        return result
    
    def bootstrap_ci(self, data: np.ndarray, ci: float = 0.95) -> Tuple[float, float]:
        """Bootstrap confidence interval for the mean."""
        means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(self.n_bootstrap)]
        alpha = 1.0 - ci
        lower = np.percentile(means, alpha / 2.0 * 100)
        upper = np.percentile(means, (1 - alpha / 2.0) * 100)
        return float(lower), float(upper)
    
    def compare_policies_full(self, 
                               metrics_a: List[Dict[str, Any]], 
                               metrics_b: List[Dict[str, Any]],
                               policy_a_name: str,
                               policy_b_name: str,
                               metric_names: List[str] = None) -> List[ComparisonResult]:
        """Compare two policies across all metrics."""
        results = []
        if not metrics_a or not metrics_b:
            return results
            
        keys_to_compare = metric_names or list(metrics_a[0].keys())
        
        for key in keys_to_compare:
            try:
                vals_a = np.array([m[key] for m in metrics_a if key in m])
                vals_b = np.array([m[key] for m in metrics_b if key in m])
                
                if len(vals_a) == 0 or len(vals_b) == 0:
                    continue
                    
                higher_is_better = not any(w in key.lower() for w in ['delay', 'latency', 'time', 'variance', 'violation', 'starvation'])
                
                res = self.compare(vals_a, vals_b, metric_name=key,
                                  policy_a_name=policy_a_name, policy_b_name=policy_b_name,
                                  higher_is_better=higher_is_better)
                results.append(res)
            except Exception:
                pass
                
        return results
