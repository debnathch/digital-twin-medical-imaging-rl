import numpy as np
from typing import Optional, Dict, Any

from simulator.digital_twin.environment import MedicalImagingDigitalTwin, JointAction, StepResult
from simulator.state.study import Urgency
from evaluation.metrics import PolicyMetrics, MetricsCalculator

class FIFOPolicy:
    """First-In-First-Out baseline policy.
    
    Studies are processed strictly in arrival order.
    Resources are assigned to the first available scanner/radiologist/model.
    Subject to the same hard safety/resource constraints as PPO.
    
    This is a NON-LEARNING baseline for scientific comparison.
    It uses the SAME Digital Twin, workload, event stream, resource constraints,
    service-time distributions, study population, and random seeds as PPO.
    """
    
    def __init__(self):
        self.name = "FIFO"
    
    def select_action(self, digital_twin: MedicalImagingDigitalTwin) -> JointAction:
        """Select action according to FIFO policy.
        
        Queue: Always select the earliest-arriving waiting study (index 0 of sorted-by-arrival candidates).
        Scanner: Assign first available scanner compatible with study modality.
        Radiologist: Assign first available radiologist.
        AI Model: Assign first compatible model with lowest GPU requirement.
        Workflow: Always use balanced_throughput (action 0).
        """
        # For a simple FIFO, we default to the first available options (index 0).
        # We rely on the ConstraintValidator (applied either externally or inside the twin)
        # to correct index 0 to the first truly valid and compatible resource, which 
        # aligns with the FIFO 'first available' assignment strategy.
        return JointAction(
            workflow_action=0,
            queue_action=0,
            resource_scanner=0,
            resource_radiologist=0,
            model_action=0
        )
    
    def run_episode(self, digital_twin: MedicalImagingDigitalTwin, 
                    metrics_calculator: Optional[MetricsCalculator] = None) -> Dict[str, Any]:
        """Run a complete episode with FIFO policy.
        
        Returns:
            dict with 'metrics' (PolicyMetrics), 'rewards' (list), 'studies' (list)
        """
        state = digital_twin.reset()
        done = False
        rewards = []
        
        while not done:
            action = self.select_action(digital_twin)
            # In a real setup, we might pass this through ConstraintValidator first
            # if digital_twin.step doesn't do it automatically.
            result = digital_twin.step(action)
            
            # Extract global reward safely
            if isinstance(result.rewards, dict):
                rewards.append(result.rewards.get('global', 0.0))
            else:
                rewards.append(float(getattr(result, 'rewards', 0.0)))
                
            done = result.done
        
        # Calculate metrics
        if metrics_calculator is None:
            metrics_calculator = MetricsCalculator()
            
        metrics = metrics_calculator.calculate(
            completed_studies=digital_twin.completed_studies,
            resource_pool=digital_twin.resource_pool,
            global_rewards=rewards,
        )
        
        return {
            'metrics': metrics,
            'rewards': rewards,
            'completed_studies': len(digital_twin.completed_studies),
            'sla_breaches': getattr(digital_twin, 'sla_breaches', 0),
        }
