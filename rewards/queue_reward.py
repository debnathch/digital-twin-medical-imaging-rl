class QueueLocalReward:
    """Local reward for the Queue Agent (Level 2).
    
    Rewards: reduced queue waiting, correct urgent prioritization, reduced SLA risk, reduced starvation.
    """
    def calculate(self,
                  queue_delay_reduction: float,    # how much delay was reduced
                  urgent_prioritized: bool,         # was urgent case correctly prioritized?
                  sla_risk_reduction: float,        # reduction in SLA risk
                  starvation_count: int,            # number of studies starved (waiting too long)
                  max_wait_time: float,             # max wait of any study (minutes)
                  ) -> float:
        """Calculate local reward for the queue agent."""
        w_delay = 0.3
        w_urgent = 0.3
        w_sla = 0.2
        w_starvation = 0.1
        w_max_wait = 0.1
        
        norm_delay = min(max(queue_delay_reduction / 30.0, 0.0), 1.0)
        norm_urgent = 1.0 if urgent_prioritized else 0.0
        norm_sla = min(max(sla_risk_reduction, 0.0), 1.0)
        penalty_starvation = min(starvation_count / 5.0, 1.0)
        penalty_max_wait = min(max_wait_time / 120.0, 1.0)
        
        reward = (
            w_delay * norm_delay +
            w_urgent * norm_urgent +
            w_sla * norm_sla -
            w_starvation * penalty_starvation -
            w_max_wait * penalty_max_wait
        )
        return reward
