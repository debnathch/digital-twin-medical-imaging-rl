import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class PolicyMetrics:
    """Complete metrics for a single policy evaluation."""
    # Primary metrics
    mean_turnaround_time: float = 0.0
    median_turnaround_time: float = 0.0
    p95_turnaround_time: float = 0.0
    mean_queue_waiting_time: float = 0.0
    p95_queue_waiting_time: float = 0.0
    sla_compliance: float = 0.0
    critical_case_waiting_time: float = 0.0
    
    # Secondary metrics
    radiologist_workload_variance: float = 0.0
    resource_utilization: float = 0.0
    scanner_utilization: float = 0.0
    gpu_utilization: float = 0.0
    ai_inference_latency: float = 0.0
    num_sla_violations: int = 0
    critical_case_starvation_count: int = 0
    throughput: float = 0.0
    completed_studies: int = 0
    rejected_actions: int = 0
    constraint_violations: int = 0
    policy_inference_time_ms: float = 0.0
    
    # Reward
    mean_global_reward: float = 0.0
    total_global_reward: float = 0.0
    
    # Confidence intervals (populated during multi-seed analysis)
    ci_95: Dict[str, tuple] = field(default_factory=dict)  # metric_name -> (lower, upper)
    
    def to_dict(self) -> Dict[str, Any]: 
        return {k: v for k, v in self.__dict__.items()}
        
    def to_row(self) -> List[Any]: 
        return list(self.to_dict().values())
    
    @staticmethod
    def column_names() -> List[str]: 
        return list(PolicyMetrics().to_dict().keys())

class MetricsCalculator:
    """Computes all 19 primary/secondary metrics from Digital Twin state and completed studies."""
    
    def calculate(self, completed_studies: list, resource_pool: Any, global_rewards: list,
                  rejected_actions: int = 0, constraint_violations: int = 0,
                  policy_inference_times: list = None) -> PolicyMetrics:
        """Calculate all metrics from simulation results.
        
        Args:
            completed_studies: list of Study objects with timing data
            resource_pool: ResourcePool with utilization data
            global_rewards: list of global rewards per step
            rejected_actions: count of rejected RL actions
            constraint_violations: count of constraint violations  
            policy_inference_times: list of inference times in ms
        """
        metrics = PolicyMetrics()
        
        if not completed_studies:
            return metrics
            
        # Extracting metrics from studies (mocking field access as object or dict)
        get_val = lambda s, k: getattr(s, k, s.get(k, 0) if isinstance(s, dict) else 0)
        
        turnaround_times = [get_val(s, 'turnaround_time') for s in completed_studies]
        turnaround_times = [t for t in turnaround_times if t is not None and t > 0]
        queue_times = [get_val(s, 'queue_waiting_time') for s in completed_studies]
        queue_times = [t for t in queue_times if t is not None and t > 0]
        
        # SLA
        n_breached = sum(1 for s in completed_studies if getattr(s, 'sla_breached', False))
        metrics.sla_compliance = 1.0 - n_breached / max(1, len(completed_studies))
        metrics.num_sla_violations = n_breached
        
        if turnaround_times:
            metrics.mean_turnaround_time = float(np.mean(turnaround_times))
            metrics.median_turnaround_time = float(np.median(turnaround_times))
            metrics.p95_turnaround_time = float(np.percentile(turnaround_times, 95))
            
        if queue_times:
            metrics.mean_queue_waiting_time = float(np.mean(queue_times))
            metrics.p95_queue_waiting_time = float(np.percentile(queue_times, 95))
            
        metrics.completed_studies = len(completed_studies)
        metrics.rejected_actions = rejected_actions
        metrics.constraint_violations = constraint_violations
        
        if global_rewards:
            metrics.mean_global_reward = float(np.mean(global_rewards))
            metrics.total_global_reward = float(np.sum(global_rewards))
            
        if policy_inference_times:
            metrics.policy_inference_time_ms = float(np.mean(policy_inference_times))
            
        # Dummy utilization based on resource pool
        if hasattr(resource_pool, 'utilization'):
            metrics.resource_utilization = resource_pool.utilization
            
        return metrics
    
    def calculate_confidence_intervals(self, metrics_list: List[PolicyMetrics]) -> PolicyMetrics:
        """Calculate 95% CI across multiple runs (seeds)."""
        if not metrics_list:
            return PolicyMetrics()
            
        avg_metrics = PolicyMetrics()
        keys = [k for k in PolicyMetrics.column_names() if k != 'ci_95']
        
        for key in keys:
            values = [getattr(m, key) for m in metrics_list]
            if not values or not isinstance(values[0], (int, float)):
                continue
                
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1) if len(values) > 1 else 0.0)
            ci_margin = 1.96 * std_val / np.sqrt(len(values)) if len(values) > 0 else 0.0
            
            setattr(avg_metrics, key, mean_val)
            avg_metrics.ci_95[key] = (mean_val - ci_margin, mean_val + ci_margin)
            
        return avg_metrics
