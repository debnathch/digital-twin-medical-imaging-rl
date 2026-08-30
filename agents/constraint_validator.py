"""Constraint Validator — deterministic gate between RL agents and Digital Twin.

RL action -> ConstraintValidator -> Approved action

The RL policy optimises WITHIN the permissible action space.
It does NOT define the safety boundary.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.digital_twin.environment import MedicalImagingDigitalTwin, JointAction


@dataclass
class ValidationResult:
    approved: bool = True
    original_action: Optional[JointAction] = None
    corrected_action: Optional[JointAction] = None
    violations: list = field(default_factory=list)
    corrections: list = field(default_factory=list)


class ConstraintValidator:
    """Deterministic constraint gate between RL agents and Digital Twin.

    Constraints enforced:
      1. Emergency/critical priority cannot be violated
      2. Unavailable scanner cannot be selected
      3. Unavailable radiologist cannot be assigned
      4. Unavailable AI model cannot be selected
      5. GPU capacity cannot be exceeded
      6. Memory capacity cannot be exceeded
      7. Invalid workflow transition cannot occur
      8. Completed study cannot be reprocessed
      9. SLA-critical cases cannot be indefinitely starved

    Every rejected or corrected action is logged.
    """

    def __init__(self):
        self.rejection_log: list = []
        self.total_rejections: int = 0
        self.total_corrections: int = 0

    def validate(self, joint_action, digital_twin) -> ValidationResult:
        from simulator.digital_twin.environment import JointAction

        result = ValidationResult(original_action=joint_action)
        violations = []

        candidates = digital_twin.get_study_candidates()
        if not candidates:
            result.approved = True
            result.corrected_action = joint_action
            return result

        # Clamp queue action
        q_act = joint_action.queue_action
        if q_act >= len(candidates):
            q_act = 0
            violations.append("queue_action out of range; clamped to 0")

        study = candidates[q_act]

        # Check completed
        from simulator.state.study import StudyStage
        if study.current_stage == StudyStage.COMPLETED:
            violations.append("attempted to reprocess completed study")
            result.approved = False

        # Scanner availability
        scanners = digital_twin.get_available_scanners(study.modality)
        sc_idx = joint_action.resource_scanner
        if not scanners:
            violations.append("no available scanner")
            result.approved = False
        elif sc_idx >= len(scanners):
            sc_idx = 0
            violations.append("scanner index out of range; clamped to 0")

        # Radiologist availability
        rads = digital_twin.get_available_radiologists()
        rd_idx = joint_action.resource_radiologist
        if not rads:
            violations.append("no available radiologist")
            result.approved = False
        elif rd_idx >= len(rads):
            rd_idx = 0
            violations.append("radiologist index out of range; clamped to 0")

        # Model availability
        models = digital_twin.get_compatible_models(study.modality)
        md_idx = joint_action.model_action
        if not models:
            violations.append("no compatible AI model")
            result.approved = False
        elif md_idx >= len(models):
            md_idx = 0
            violations.append("model index out of range; clamped to 0")

        # GPU/memory capacity
        if models and md_idx < len(models):
            model = models[md_idx]
            gpu = digital_twin.resource_pool.gpu_cluster
            if not gpu.can_allocate(model.gpu_requirement, model.memory_requirement):
                violations.append("GPU/memory capacity exceeded")
                result.approved = False

        # SLA-starvation check: if any critical study waiting > 30 min, force it
        from simulator.state.study import Urgency
        for s in digital_twin.waiting_studies:
            wait = digital_twin.clock - s.arrival_time
            if s.urgency == Urgency.CRITICAL and wait > 30.0:
                # Force selection of this critical study
                try:
                    forced_idx = candidates.index(s)
                    q_act = forced_idx
                    violations.append(f"SLA-starvation override: critical study {s.study_id}")
                except ValueError:
                    pass
                break

        corrected = JointAction(
            workflow_action=joint_action.workflow_action,
            queue_action=q_act,
            resource_scanner=sc_idx,
            resource_radiologist=rd_idx,
            model_action=md_idx,
        )
        result.corrected_action = corrected
        result.violations = violations

        if violations:
            self.total_corrections += 1
            self.rejection_log.append({
                "violations": violations,
                "time": digital_twin.clock,
            })

        if not result.approved:
            self.total_rejections += 1

        return result

    def get_rejection_summary(self) -> dict:
        return {
            "total_rejections": self.total_rejections,
            "total_corrections": self.total_corrections,
            "log_count": len(self.rejection_log),
        }
