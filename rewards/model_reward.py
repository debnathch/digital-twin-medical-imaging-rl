class ModelLocalReward:
    """Local reward for the AI Model Agent (Level 2).
    
    Rewards: model suitability, low inference latency, feasible compute, avoidance of unnecessary high-cost models.
    """
    def calculate(self,
                  model_suitability: float,        # 0-1 match score
                  inference_time: float,            # minutes
                  compute_feasible: bool,           # GPU/memory within capacity?
                  model_cost: float,                # relative cost 0-1
                  ) -> float:
        """Calculate local reward for the model agent."""
        if not compute_feasible:
            return -1.0 # Invalid action penalty
            
        w_suitability = 0.5
        w_latency = 0.3
        w_cost = 0.2
        
        penalty_latency = min(inference_time / 30.0, 1.0)
        
        reward = (
            w_suitability * model_suitability -
            w_latency * penalty_latency -
            w_cost * model_cost
        )
        return reward
