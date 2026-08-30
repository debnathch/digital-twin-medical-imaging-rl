"""Static Triage Priority baseline policy.

Fixed priority ordering: Critical > Urgent > Routine.
Within each priority class, use FIFO (arrival order).

This is a NON-LEARNING baseline for scientific comparison.
Uses the SAME Digital Twin, workload, seeds, and constraints as PPO.
"""

from simulator.digital_twin.environment import MedicalImagingDigitalTwin, JointAction
from simulator.state.study import Urgency
from evaluation.metrics import MetricsCalculator


class TriagePriorityPolicy:
    """Static Triage Priority baseline.

    Critical > Urgent > Routine, FIFO within each class.
    """

    def __init__(self):
        self.name = "STATIC_TRIAGE"

    def select_action(self, digital_twin: MedicalImagingDigitalTwin) -> JointAction:
        candidates = digital_twin.get_study_candidates()
        if not candidates:
            return JointAction()

        # Sort: Critical (0) < Urgent (1) < Routine (2), then by arrival_time
        sorted_cands = sorted(candidates, key=lambda s: (s.urgency.value, s.arrival_time))
        q_action = 0  # pick first after triage sort
        # Remap to index in original candidates list
        try:
            q_action = candidates.index(sorted_cands[0])
        except (ValueError, IndexError):
            q_action = 0

        study = sorted_cands[0]

        # Scanner: first available compatible
        scanners = digital_twin.get_available_scanners(study.modality)
        sc_idx = 0

        # Radiologist: first available
        rads = digital_twin.get_available_radiologists()
        rd_idx = 0

        # Model: highest suitability for this modality
        models = digital_twin.get_compatible_models(study.modality)
        md_idx = 0
        if models:
            scores = [m.suitability_score(study.modality, study.body_region) for m in models]
            md_idx = int(max(range(len(scores)), key=lambda i: scores[i]))

        # Workflow: urgent_priority when critical cases present, else balanced
        has_critical = any(s.urgency == Urgency.CRITICAL for s in candidates)
        wf = 1 if has_critical else 0

        return JointAction(
            workflow_action=wf,
            queue_action=q_action,
            resource_scanner=sc_idx,
            resource_radiologist=rd_idx,
            model_action=md_idx,
        )

    def run_episode(self, digital_twin: MedicalImagingDigitalTwin,
                    metrics_calculator: MetricsCalculator = None) -> dict:
        state = digital_twin.reset()
        done = False
        rewards = []

        while not done:
            action = self.select_action(digital_twin)
            result = digital_twin.step(action)
            rewards.append(result.rewards.get("global", 0.0))
            done = result.done

        if metrics_calculator is None:
            metrics_calculator = MetricsCalculator()
        metrics = metrics_calculator.calculate(
            completed_studies=digital_twin.completed_studies,
            resource_pool=digital_twin.resource_pool,
            global_rewards=rewards,
        )
        return {
            "metrics": metrics,
            "rewards": rewards,
            "completed_studies": len(digital_twin.completed_studies),
            "sla_breaches": digital_twin.sla_breaches,
        }
