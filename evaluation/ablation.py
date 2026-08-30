from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class AblationVariant:
    name: str
    description: str
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    # What to disable/modify for this variant

@dataclass
class AblationResult:
    variant_name: str
    metrics: Dict[str, Any] = field(default_factory=dict)  # PolicyMetrics as dict
    comparison_to_full: Dict[str, Any] = field(default_factory=dict)  # ComparisonResult as dict
    degradation_percentage: float = 0.0

class AblationRunner:
    """Runs ablation studies to validate architectural contributions.
    
    Variants:
    A. Full hierarchical PPO/HAPPO (baseline)
    B. PPO without hierarchy (flat single-agent PPO)
    C. PPO without global reward (only local rewards)
    D. PPO without resource state (remove resource observations)
    E. PPO without model state (remove model observations)
    F. PPO without Workflow Agent (only Level 2 agents)
    """
    
    VARIANTS = [
        AblationVariant('full', 'Full hierarchical PPO/HAPPO'),
        AblationVariant('no_hierarchy', 'PPO without hierarchy (flat single-agent)', 
                       config_overrides={'flatten_hierarchy': True}),
        AblationVariant('no_global_reward', 'PPO without global reward',
                       config_overrides={'beta_weight': 0.0}),
        AblationVariant('no_resource_state', 'PPO without resource state',
                       config_overrides={'mask_resource_obs': True}),
        AblationVariant('no_model_state', 'PPO without model state',
                       config_overrides={'mask_model_obs': True}),
        AblationVariant('no_workflow_agent', 'PPO without Workflow Agent',
                       config_overrides={'disable_workflow_agent': True}),
    ]
    
    def get_variants(self) -> List[AblationVariant]:
        return self.VARIANTS
    
    def compare_results(self, full_metrics: Dict[str, Any], variant_metrics: Dict[str, Any], 
                       variant_name: str) -> AblationResult:
        """Compare variant against full model."""
        result = AblationResult(variant_name=variant_name)
        result.metrics = variant_metrics
        
        full_val = full_metrics.get('mean_global_reward', 1.0)
        var_val = variant_metrics.get('mean_global_reward', 1.0)
        
        # Calculate degradation (how much worse the variant is compared to full)
        if full_val != 0:
            result.degradation_percentage = (full_val - var_val) / abs(full_val) * 100.0
            
        result.comparison_to_full = {
            'metric': 'mean_global_reward',
            'full_score': full_val,
            'variant_score': var_val,
            'diff': full_val - var_val
        }
        
        return result
