import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class RewardBreakdown:
    """Detailed breakdown of every reward component for explainability."""
    # Raw values (before normalization)
    raw_sla_compliance: float = 0.0
    raw_clinical_suitability: float = 0.0
    raw_queue_delay: float = 0.0
    raw_inference_latency: float = 0.0
    raw_report_delay: float = 0.0
    raw_resource_inefficiency: float = 0.0
    raw_workload_imbalance: float = 0.0
    
    # Normalized values [0, 1]
    norm_sla_compliance: float = 0.0
    norm_clinical_suitability: float = 0.0
    norm_queue_delay: float = 0.0
    norm_inference_latency: float = 0.0
    norm_report_delay: float = 0.0
    norm_resource_inefficiency: float = 0.0
    norm_workload_imbalance: float = 0.0
    
    # Weighted contributions
    weighted_sla: float = 0.0
    weighted_clinical: float = 0.0
    weighted_queue_delay: float = 0.0
    weighted_inference: float = 0.0
    weighted_report: float = 0.0
    weighted_resource: float = 0.0
    weighted_workload: float = 0.0
    
    # Final
    global_reward: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging and UI display."""
        return {k: v for k, v in self.__dict__.items()}

@dataclass
class RewardWeights:
    """Configurable reward weights.
    R_G = w1*C_SLA + w2*C_Clinical - w3*T_Queue - w4*T_Inference - w5*T_Report - w6*U_Resource - w7*I_Workload
    """
    w1_sla: float = 0.25
    w2_clinical: float = 0.15
    w3_queue_delay: float = 0.15
    w4_inference: float = 0.10
    w5_report: float = 0.15
    w6_resource: float = 0.10
    w7_workload: float = 0.10

class GlobalRewardCalculator:
    """Calculates the global reward R_G with full normalization.
    
    R_G = w1*C_SLA + w2*C_Clinical - w3*T_Queue - w4*T_Inference - w5*T_Report - w6*U_Resource - w7*I_Workload
    
    All metrics are normalized to [0, 1] before weighting to prevent any single metric from dominating.
    """
    
    # Normalization constants (configurable)
    MAX_QUEUE_DELAY_MINUTES = 120.0
    MAX_INFERENCE_LATENCY_MINUTES = 30.0
    MAX_REPORT_DELAY_MINUTES = 240.0
    
    def __init__(self, weights: Optional[RewardWeights] = None):
        self.weights = weights or RewardWeights()
    
    def calculate(self, 
                  sla_compliance_rate: float,        # 0-1
                  clinical_suitability: float,        # 0-1
                  mean_queue_delay: float,            # minutes
                  mean_inference_latency: float,      # minutes
                  mean_report_delay: float,           # minutes
                  resource_utilization: float,         # 0-1 (higher is better, but too high is bad)
                  workload_imbalance: float,          # coefficient of variation, 0+ (lower is better)
                  ) -> RewardBreakdown:
        """Calculate global reward with full normalization and breakdown."""
        # 1. Store raw values
        raw_resource_inefficiency = abs(resource_utilization - 0.75) / 0.75 if resource_utilization else 1.0
        
        breakdown = RewardBreakdown(
            raw_sla_compliance=sla_compliance_rate,
            raw_clinical_suitability=clinical_suitability,
            raw_queue_delay=mean_queue_delay,
            raw_inference_latency=mean_inference_latency,
            raw_report_delay=mean_report_delay,
            raw_resource_inefficiency=raw_resource_inefficiency,
            raw_workload_imbalance=workload_imbalance
        )
        
        # 2. Normalize all to [0, 1]
        breakdown.norm_sla_compliance = np.clip(sla_compliance_rate, 0.0, 1.0)
        breakdown.norm_clinical_suitability = np.clip(clinical_suitability, 0.0, 1.0)
        breakdown.norm_queue_delay = np.clip(mean_queue_delay / self.MAX_QUEUE_DELAY_MINUTES, 0.0, 1.0)
        breakdown.norm_inference_latency = np.clip(mean_inference_latency / self.MAX_INFERENCE_LATENCY_MINUTES, 0.0, 1.0)
        breakdown.norm_report_delay = np.clip(mean_report_delay / self.MAX_REPORT_DELAY_MINUTES, 0.0, 1.0)
        breakdown.norm_resource_inefficiency = np.clip(raw_resource_inefficiency, 0.0, 1.0)
        breakdown.norm_workload_imbalance = np.clip(workload_imbalance, 0.0, 1.0)
        
        # 3. Apply weights
        breakdown.weighted_sla = self.weights.w1_sla * breakdown.norm_sla_compliance
        breakdown.weighted_clinical = self.weights.w2_clinical * breakdown.norm_clinical_suitability
        breakdown.weighted_queue_delay = self.weights.w3_queue_delay * breakdown.norm_queue_delay
        breakdown.weighted_inference = self.weights.w4_inference * breakdown.norm_inference_latency
        breakdown.weighted_report = self.weights.w5_report * breakdown.norm_report_delay
        breakdown.weighted_resource = self.weights.w6_resource * breakdown.norm_resource_inefficiency
        breakdown.weighted_workload = self.weights.w7_workload * breakdown.norm_workload_imbalance
        
        # 4. Final reward calculation
        breakdown.global_reward = (
            breakdown.weighted_sla + 
            breakdown.weighted_clinical - 
            breakdown.weighted_queue_delay - 
            breakdown.weighted_inference - 
            breakdown.weighted_report - 
            breakdown.weighted_resource - 
            breakdown.weighted_workload
        )
        
        return breakdown
