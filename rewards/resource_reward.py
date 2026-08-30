class ResourceLocalReward:
    """Local reward for the Resource Agent (Level 2).
    
    Rewards: feasible allocation, reduced idle capacity, balanced radiologist workload, balanced scanner usage.
    """
    def calculate(self,
                  allocation_feasible: bool,        # was the allocation valid?
                  scanner_utilization: float,       # 0-1
                  radiologist_utilization: float,   # 0-1
                  workload_variance: float,         # variance across radiologists
                  idle_resources: int,              # number of idle resources
                  ) -> float:
        """Calculate local reward for the resource agent."""
        if not allocation_feasible:
            return -1.0 # Severe penalty for invalid allocation
            
        w_scanner = 0.3
        w_radiologist = 0.3
        w_variance = 0.2
        w_idle = 0.2
        
        # We want optimal utilization, say 75%
        scanner_score = 1.0 - abs(scanner_utilization - 0.75) / 0.75
        scanner_score = max(scanner_score, 0.0)
        
        radiologist_score = 1.0 - abs(radiologist_utilization - 0.75) / 0.75
        radiologist_score = max(radiologist_score, 0.0)
        
        penalty_variance = min(workload_variance, 1.0)
        penalty_idle = min(idle_resources / 10.0, 1.0)
        
        reward = (
            w_scanner * scanner_score +
            w_radiologist * radiologist_score -
            w_variance * penalty_variance -
            w_idle * penalty_idle
        )
        return reward
