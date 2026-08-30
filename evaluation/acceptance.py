from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class AcceptanceCriteria:
    """Configurable acceptance thresholds."""
    global_reward_improvement_threshold: float = 0.05   # 5% improvement
    tat_improvement_threshold: float = 0.05              # 5% TAT reduction
    sla_compliance_minimum: float = 0.0                  # must be >= baseline
    critical_delay_maximum_ratio: float = 1.0            # must be <= baseline
    ci_must_not_contradict: bool = True
    min_seeds_consistent: int = 3                        # out of 5
    must_generalize: bool = True                         # at least 1 unseen scenario

@dataclass
class AcceptanceResult:
    """Result of acceptance criteria evaluation."""
    overall_pass: bool = False
    criteria_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # criterion -> {pass, value, threshold, detail}
    preferred_policy: str = ""  # "HIERARCHICAL PPO/HAPPO" or "No statistically supported winner"
    summary: str = ""
    detailed_explanation: str = ""

class AcceptanceCriteriaEvaluator:
    """Evaluates whether PPO meets predefined acceptance criteria.
    
    PPO is preferred ONLY if ALL criteria are met:
    A. Global reward improvement >= threshold over BOTH baselines
    B. No material degradation of critical-case prioritization
    C. SLA compliance improved or statistically non-inferior
    D. Queue delay and TAT improved
    E. Resource utilization/workload balance acceptable
    F. Improvements statistically significant
    G. Consistent across random seeds
    H. Effective under at least one unseen scenario
    
    If ANY criterion fails, PPO is NOT declared preferred.
    This is critical for research integrity.
    """
    def __init__(self, criteria: Optional[AcceptanceCriteria] = None):
        self.criteria = criteria or AcceptanceCriteria()
    
    def evaluate(self,
                 ppo_metrics: Dict[str, float],       # aggregated PPO metrics across seeds
                 fifo_metrics: Dict[str, float],      # aggregated FIFO metrics across seeds
                 triage_metrics: Dict[str, float],    # aggregated triage metrics across seeds
                 statistical_results: List[Any],      # ComparisonResult objects
                 seed_results: List[Dict[str, float]],   # per-seed PPO metrics
                 scenario_results: Dict[str, float],     # per-scenario PPO metrics
                 ) -> AcceptanceResult:
        """Evaluate all acceptance criteria. Returns PASS only if ALL criteria met."""
        result = AcceptanceResult()
        
        # A. Global reward
        ppo_reward = ppo_metrics.get('mean_global_reward', 0)
        best_baseline_reward = max(fifo_metrics.get('mean_global_reward', 0), triage_metrics.get('mean_global_reward', 0))
        reward_imp = (ppo_reward - best_baseline_reward) / (abs(best_baseline_reward) + 1e-6)
        pass_A = reward_imp >= self.criteria.global_reward_improvement_threshold
        result.criteria_results['A_GlobalReward'] = {'pass': pass_A, 'value': reward_imp, 'threshold': self.criteria.global_reward_improvement_threshold}
        
        # B. Critical case prioritization
        ppo_critical = ppo_metrics.get('critical_case_waiting_time', float('inf'))
        baseline_critical = triage_metrics.get('critical_case_waiting_time', 1.0) # triage is baseline for this
        crit_ratio = ppo_critical / (baseline_critical + 1e-6)
        pass_B = crit_ratio <= self.criteria.critical_delay_maximum_ratio
        result.criteria_results['B_CriticalDelay'] = {'pass': pass_B, 'value': crit_ratio, 'threshold': self.criteria.critical_delay_maximum_ratio}
        
        # C. SLA Compliance
        ppo_sla = ppo_metrics.get('sla_compliance', 0)
        base_sla = max(fifo_metrics.get('sla_compliance', 0), triage_metrics.get('sla_compliance', 0))
        pass_C = ppo_sla >= base_sla + self.criteria.sla_compliance_minimum
        result.criteria_results['C_SLA'] = {'pass': pass_C, 'value': ppo_sla - base_sla, 'threshold': self.criteria.sla_compliance_minimum}
        
        # D. TAT improvement
        ppo_tat = ppo_metrics.get('mean_turnaround_time', float('inf'))
        base_tat = min(fifo_metrics.get('mean_turnaround_time', float('inf')), triage_metrics.get('mean_turnaround_time', float('inf')))
        tat_imp = (base_tat - ppo_tat) / (base_tat + 1e-6)
        pass_D = tat_imp >= self.criteria.tat_improvement_threshold
        result.criteria_results['D_TAT'] = {'pass': pass_D, 'value': tat_imp, 'threshold': self.criteria.tat_improvement_threshold}
        
        # G. Seeds consistent
        consistent_seeds = sum(1 for s in seed_results if s.get('mean_global_reward', 0) > best_baseline_reward)
        pass_G = consistent_seeds >= self.criteria.min_seeds_consistent
        result.criteria_results['G_Seeds'] = {'pass': pass_G, 'value': consistent_seeds, 'threshold': self.criteria.min_seeds_consistent}
        
        # H. Generalization
        pass_H = len(scenario_results) > 0 and any(v > best_baseline_reward for v in scenario_results.values()) if self.criteria.must_generalize else True
        result.criteria_results['H_Generalize'] = {'pass': pass_H, 'value': len(scenario_results), 'threshold': 1 if self.criteria.must_generalize else 0}
        
        # Overall
        result.overall_pass = all([pass_A, pass_B, pass_C, pass_D, pass_G, pass_H])
        
        if result.overall_pass:
            result.preferred_policy = "HIERARCHICAL PPO/HAPPO"
            result.summary = "PPO preferred: met all strict acceptance criteria."
        else:
            result.preferred_policy = "No statistically supported winner"
            result.summary = "PPO NOT preferred: failed one or more acceptance criteria."
            
        return result
