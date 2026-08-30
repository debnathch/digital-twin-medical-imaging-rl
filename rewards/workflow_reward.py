class WorkflowLocalReward:
    """Local reward for the Workflow Agent (Level 1).
    
    Rewards high-level workflow strategy decisions that improve overall system performance.
    Considers: system throughput, balanced processing, SLA management.
    """
    def calculate(self,
                  throughput_rate: float,        # studies completed per hour
                  processing_balance: float,     # balance across stages (0-1)
                  sla_compliance: float,         # 0-1
                  strategy_consistency: float,   # did the strategy help? (0-1)
                  ) -> float:
        """Returns local reward for workflow agent."""
        # Normalize and weight the components
        # A simple linear combination
        w_throughput = 0.3
        w_balance = 0.2
        w_sla = 0.4
        w_consistency = 0.1
        
        # Assuming max practical throughput is say 100 per hour
        norm_throughput = min(throughput_rate / 100.0, 1.0)
        
        reward = (
            w_throughput * norm_throughput +
            w_balance * processing_balance +
            w_sla * sla_compliance +
            w_consistency * strategy_consistency
        )
        return reward
