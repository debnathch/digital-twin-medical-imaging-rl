"""Medical Imaging Digital Twin — Event-Driven Simulation Environment.

This is the ENVIRONMENT in the RL formulation. NOT an agent.
It simulates a hospital medical imaging workflow with:
- Study arrivals (Poisson process)
- Scanner acquisition
- AI-assisted analysis
- Radiologist reporting
- SLA tracking

SYNTHETIC / RESEARCH DATA ONLY
"""

import numpy as np
import random
from typing import Optional, List, Dict
from dataclasses import dataclass, field
import uuid

from simulator.events.event import Event, EventType
from simulator.events.event_queue import EventQueue
from simulator.events.event_sources import MockEventSource
from simulator.state.study import Study, Urgency, Modality, StudyStage
from simulator.state.global_state import (
    GlobalState, QueueState, WorkflowState, ModelState,
    ComputeState, HumanResourceState, SLAState, ClinicalContext,
)
from simulator.resources.resources import (
    Scanner, Radiologist, AIModel, GPUCluster, ResourcePool,
)
from simulator.scenarios.scenario import ScenarioConfig


@dataclass
class JointAction:
    """Combined action from all agents."""
    workflow_action: int = 0       # 0-4: strategy
    queue_action: int = 0          # index into candidate studies
    resource_scanner: int = 0      # scanner index
    resource_radiologist: int = 0  # radiologist index
    model_action: int = 0          # AI model index


@dataclass
class StepResult:
    """Result of one simulation step."""
    global_state: GlobalState = None
    rewards: dict = field(default_factory=dict)
    done: bool = False
    info: dict = field(default_factory=dict)
    events_processed: list = field(default_factory=list)
    constraint_violations: list = field(default_factory=list)


# ── Normalization constants ──────────────────────────────────────────
_MAX_QUEUE = 200
_MAX_WAIT_MIN = 1440.0  # 24 h
_MAX_STUDIES = 300
_MAX_COMPLEXITY = 5.0


class MedicalImagingDigitalTwin:
    """Stateful discrete-event medical imaging workflow simulator.

    The Digital Twin maintains: simulation clock, event queue, study queue,
    scanner/radiologist/GPU resources, SLA tracking, workflow state.
    """

    def __init__(self, config: dict, scenario: ScenarioConfig, seed: int = 42):
        self.raw_config = config
        self.scenario = scenario
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

        # Episode limits
        self.max_steps = config.get("max_steps_per_episode", 200)
        self.max_studies = scenario.studies_per_episode

        # State initialised in reset()
        self.clock: float = 0.0
        self.event_queue = EventQueue()
        self.resource_pool: Optional[ResourcePool] = None

        self.waiting_studies: List[Study] = []
        self.active_studies: List[Study] = []
        self.completed_studies: List[Study] = []
        self.all_studies: Dict[str, Study] = {}

        self.total_steps = 0
        self.sla_breaches = 0
        self.rejected_actions = 0
        self.constraint_violations_count = 0
        self.studies_generated = 0

        self.event_source = MockEventSource()

    # ── reset / step ──────────────────────────────────────────────────

    def reset(self) -> GlobalState:
        """Reset simulation and return the initial global state."""
        self.rng = np.random.default_rng(self.seed)
        self.py_rng = random.Random(self.seed)

        self.clock = 0.0
        self.event_queue = EventQueue()

        self.waiting_studies = []
        self.active_studies = []
        self.completed_studies = []
        self.all_studies = {}

        self.total_steps = 0
        self.sla_breaches = 0
        self.rejected_actions = 0
        self.constraint_violations_count = 0
        self.studies_generated = 0

        # Create resources from scenario
        rc = {
            "num_scanners": self.scenario.num_scanners,
            "num_radiologists": int(
                self.scenario.num_radiologists
                * self.scenario.radiologist_availability_fraction
            ),
            "gpu_capacity_fraction": self.scenario.gpu_capacity_fraction,
        }
        self.resource_pool = ResourcePool.create_default(rc)

        # Generate initial study-arrival events
        duration_min = self.scenario.simulation_duration_hours * 60.0
        events = self.event_source.generate_events(
            self.rng, 0.0, duration_min, self.scenario
        )
        for ev in events[: self.max_studies]:
            self.event_queue.schedule(ev)
            self.studies_generated += 1

        # Process arrivals at t=0 so the queue is not empty
        self._process_events_up_to(0.0)
        return self.get_global_state()

    def step(self, action: JointAction) -> StepResult:
        """Execute one RL decision step.

        Each step covers a decision interval (~6-15 min depending on load).
        1. Advance clock by decision_interval.
        2. Process all events up to new clock time (arrivals, completions).
        3. Apply joint action to assign a waiting study.
        4. Return new state, done flag.
        """
        events_processed: List[Event] = []
        violations: List[str] = []

        # Decision interval: spread total sim time across max_steps
        sim_duration = self.scenario.simulation_duration_hours * 60.0
        decision_interval = sim_duration / max(1, self.max_steps)

        # 1. Advance clock
        self.clock += decision_interval

        # 2. Process ALL events up to new clock time
        self._process_events_up_to(self.clock)

        # 3. Apply joint action — try to assign one waiting study
        action_info = self._apply_action(action, violations)

        # 4. Process any immediate completion events generated by the action
        self._process_events_up_to(self.clock)

        self.total_steps += 1

        # 4. Build state
        state = self.get_global_state()
        done = self._check_done()

        return StepResult(
            global_state=state,
            rewards={"global": 0.0},  # Rewards computed externally
            done=done,
            info=action_info,
            events_processed=events_processed,
            constraint_violations=violations,
        )

    # ── Event processing ──────────────────────────────────────────────

    def _process_events_up_to(self, up_to_time: float):
        """Process all events with timestamp <= up_to_time."""
        while not self.event_queue.is_empty():
            nxt = self.event_queue.peek()
            if nxt.timestamp > up_to_time:
                break
            ev = self.event_queue.next_event()
            self.clock = max(self.clock, ev.timestamp)
            self._handle_event(ev)

    def _handle_event(self, event: Event):
        handlers = {
            EventType.STUDY_ARRIVAL: self._on_study_arrival,
            EventType.SCAN_COMPLETED: self._on_scan_completed,
            EventType.AI_INFERENCE_COMPLETED: self._on_ai_completed,
            EventType.REPORT_COMPLETED: self._on_report_completed,
            EventType.SLA_BREACH: self._on_sla_breach,
            EventType.RESOURCE_AVAILABLE: self._on_resource_available,
        }
        h = handlers.get(event.event_type)
        if h:
            h(event)

    def _on_study_arrival(self, event: Event):
        u_map = {"CRITICAL": Urgency.CRITICAL, "URGENT": Urgency.URGENT, "ROUTINE": Urgency.ROUTINE}
        urgency = u_map.get(event.metadata.get("urgency", "ROUTINE"), Urgency.ROUTINE)
        modality = Modality(event.metadata.get("modality", "CT"))
        study = Study(
            study_id=event.study_id or str(uuid.uuid4()),
            modality=modality,
            body_region=event.metadata.get("body_region", "chest"),
            urgency=urgency,
            complexity=int(event.metadata.get("complexity", 3)),
            arrival_time=event.timestamp,
            sla_deadline=event.timestamp + Study.get_sla_hours(urgency) * 60.0,
        )
        self.all_studies[study.study_id] = study
        self.waiting_studies.append(study)

        # Schedule SLA-breach warning
        self.event_queue.schedule(
            Event.create(EventType.SLA_BREACH, study.sla_deadline, study.study_id)
        )

    def _on_scan_completed(self, event: Event):
        study = self.all_studies.get(event.study_id)
        if study and study.current_stage == StudyStage.SCANNING:
            study.current_stage = StudyStage.AI_PROCESSING
            study.scan_end_time = self.clock
            self._release_scanner(study.assigned_scanner)

    def _on_ai_completed(self, event: Event):
        study = self.all_studies.get(event.study_id)
        if study and study.current_stage == StudyStage.AI_PROCESSING:
            study.current_stage = StudyStage.REPORTING
            study.ai_end_time = self.clock
            self._release_model(study.assigned_model)

    def _on_report_completed(self, event: Event):
        study = self.all_studies.get(event.study_id)
        if study and study.current_stage == StudyStage.REPORTING:
            study.current_stage = StudyStage.COMPLETED
            study.report_end_time = self.clock
            study.completion_time = self.clock
            self._release_radiologist(study.assigned_radiologist, study)
            if study in self.active_studies:
                self.active_studies.remove(study)
            self.completed_studies.append(study)

    def _on_sla_breach(self, event: Event):
        study = self.all_studies.get(event.study_id)
        if study and study.current_stage != StudyStage.COMPLETED and not study.sla_breached:
            study.sla_breached = True
            self.sla_breaches += 1

    def _on_resource_available(self, event: Event):
        pass  # Resources freed in specific handlers

    # ── Action application ────────────────────────────────────────────

    def _apply_action(self, action: JointAction, violations: list) -> dict:
        """Apply joint action: assign study → scanner → AI model → radiologist."""
        info: Dict = {"action_applied": False, "reason": ""}

        # Pick study from waiting queue
        candidates = self.get_study_candidates()
        if not candidates:
            info["reason"] = "no_waiting_studies"
            return info

        idx = min(action.queue_action, len(candidates) - 1)
        study = candidates[idx]

        # Find scanner
        scanners = self.get_available_scanners(study.modality)
        if not scanners:
            info["reason"] = "no_available_scanner"
            return info
        sc_idx = min(action.resource_scanner, len(scanners) - 1)
        scanner = scanners[sc_idx]

        # Find radiologist
        rads = self.get_available_radiologists()
        if not rads:
            info["reason"] = "no_available_radiologist"
            return info
        rd_idx = min(action.resource_radiologist, len(rads) - 1)
        rad = rads[rd_idx]

        # Find AI model
        models = self.get_compatible_models(study.modality)
        if not models:
            info["reason"] = "no_compatible_model"
            return info
        md_idx = min(action.model_action, len(models) - 1)
        model = models[md_idx]

        # Check GPU capacity
        if not self.resource_pool.gpu_cluster.can_allocate(
            model.gpu_requirement, model.memory_requirement
        ):
            info["reason"] = "gpu_capacity_exceeded"
            violations.append("GPU capacity exceeded")
            self.constraint_violations_count += 1
            return info

        # ── All checks passed — execute assignment ──
        self.waiting_studies.remove(study)
        self.active_studies.append(study)
        info["action_applied"] = True
        info["study_id"] = study.study_id

        # Start scanning
        study.current_stage = StudyStage.SCANNING
        study.assigned_scanner = scanner.scanner_id
        scanner.is_available = False
        scanner.current_study_id = study.study_id
        study.scan_start_time = self.clock

        scan_dur = self._scan_duration(study.modality, study.complexity)
        scanner.total_scan_time += scan_dur
        scanner.total_scans += 1
        self.event_queue.schedule(
            Event.create(EventType.SCAN_COMPLETED, self.clock + scan_dur, study.study_id)
        )

        # Schedule AI inference to start right after scan
        ai_start = self.clock + scan_dur
        study.assigned_model = model.model_id
        model.is_available = False
        model.current_study_id = study.study_id
        self.resource_pool.gpu_cluster.allocate(model.gpu_requirement, model.memory_requirement)
        study.ai_start_time = ai_start
        study.current_stage = StudyStage.SCANNING  # still scanning
        ai_dur = self._inference_duration(model, study.complexity)
        self.event_queue.schedule(
            Event.create(EventType.AI_INFERENCE_COMPLETED, ai_start + ai_dur, study.study_id)
        )

        # Transition scan_complete -> AI processing happens in event handler
        # Here we just schedule the cascade:
        # scan done -> study becomes SCAN_COMPLETE (event handler)
        # AI done   -> study becomes AI_COMPLETE (event handler)
        # Then we need to schedule report start after AI

        report_start = ai_start + ai_dur
        study.assigned_radiologist = rad.radiologist_id
        rad.is_available = False
        rad.current_study_id = study.study_id
        study.report_start_time = report_start
        report_dur = self._report_duration(study.modality, study.complexity)
        rad.total_report_time += report_dur
        self.event_queue.schedule(
            Event.create(EventType.REPORT_COMPLETED, report_start + report_dur, study.study_id)
        )

        return info

    # ── Timing helpers ────────────────────────────────────────────────

    def _scan_duration(self, modality: Modality, complexity: int) -> float:
        base = {Modality.CT: (10, 30), Modality.MRI: (30, 90),
                Modality.XR: (5, 15), Modality.US: (15, 45)}
        lo, hi = base[modality]
        mean = lo + (hi - lo) * (complexity / _MAX_COMPLEXITY)
        return max(1.0, float(self.rng.normal(mean, mean * 0.1)))

    def _inference_duration(self, model: AIModel, complexity: int) -> float:
        base = model.base_inference_time * (1 + complexity * 0.2)
        return max(0.5, base * self.scenario.inference_latency_multiplier)

    def _report_duration(self, modality: Modality, complexity: int) -> float:
        base = {Modality.CT: (15, 40), Modality.MRI: (20, 45),
                Modality.XR: (5, 15), Modality.US: (10, 25)}
        lo, hi = base[modality]
        mean = lo + (hi - lo) * (complexity / _MAX_COMPLEXITY)
        return max(1.0, float(self.rng.normal(mean, mean * 0.15)))

    # ── Resource release helpers ──────────────────────────────────────

    def _release_scanner(self, scanner_id: Optional[str]):
        if not scanner_id:
            return
        for s in self.resource_pool.scanners:
            if s.scanner_id == scanner_id:
                s.is_available = True
                s.current_study_id = None
                return

    def _release_model(self, model_id: Optional[str]):
        if not model_id:
            return
        for m in self.resource_pool.ai_models:
            if m.model_id == model_id:
                m.is_available = True
                m.current_study_id = None
                self.resource_pool.gpu_cluster.deallocate(
                    m.gpu_requirement, m.memory_requirement
                )
                return

    def _release_radiologist(self, rad_id: Optional[str], study: Study = None):
        if not rad_id:
            return
        for r in self.resource_pool.radiologists:
            if r.radiologist_id == rad_id:
                r.is_available = True
                r.current_study_id = None
                r.completed_reports += 1
                return

    # ── Query helpers ─────────────────────────────────────────────────

    def get_study_candidates(self, max_candidates: int = 8) -> List[Study]:
        """Return top-K waiting studies sorted by urgency then arrival."""
        sorted_studies = sorted(
            self.waiting_studies,
            key=lambda s: (s.urgency.value, s.arrival_time),
        )
        return sorted_studies[:max_candidates]

    def get_available_scanners(self, modality: Modality = None) -> List[Scanner]:
        return [
            s for s in self.resource_pool.scanners
            if s.is_available and (modality is None or s.modality == modality)
        ]

    def get_available_radiologists(self) -> List[Radiologist]:
        return [r for r in self.resource_pool.radiologists if r.is_available]

    def get_compatible_models(self, modality: Modality) -> List[AIModel]:
        return [
            m for m in self.resource_pool.ai_models
            if m.is_available and modality in m.supported_modalities
        ]

    # ── Global state builder ──────────────────────────────────────────

    def get_global_state(self) -> GlobalState:
        """Build complete normalised global state S_t."""
        waiting = self.waiting_studies
        completed = self.completed_studies
        all_studies = list(self.all_studies.values())

        # Queue state
        n_waiting = len(waiting)
        urgent_waiting = sum(1 for s in waiting if s.urgency.value <= 1)
        waits = [self.clock - s.arrival_time for s in waiting] if waiting else [0.0]
        at_risk = sum(
            1 for s in waiting
            if s.sla_deadline > 0 and (self.clock / s.sla_deadline) > 0.8
        )

        queue_st = QueueState(
            queue_length=min(1.0, n_waiting / _MAX_QUEUE),
            urgent_queue_length=min(1.0, urgent_waiting / max(1, _MAX_QUEUE)),
            max_wait=min(1.0, max(waits) / _MAX_WAIT_MIN),
            mean_wait=min(1.0, float(np.mean(waits)) / _MAX_WAIT_MIN),
            priority_distribution=[
                sum(1 for s in waiting if s.urgency == Urgency.CRITICAL) / max(1, n_waiting),
                sum(1 for s in waiting if s.urgency == Urgency.URGENT) / max(1, n_waiting),
                sum(1 for s in waiting if s.urgency == Urgency.ROUTINE) / max(1, n_waiting),
            ],
            sla_risk=min(1.0, at_risk / max(1, n_waiting)),
        )

        # Workflow state
        n_scan = sum(1 for s in self.active_studies if s.current_stage == StudyStage.SCANNING)
        n_ai = sum(1 for s in self.active_studies if s.current_stage in (StudyStage.SCAN_COMPLETE, StudyStage.AI_PROCESSING))
        n_rep = sum(1 for s in self.active_studies if s.current_stage in (StudyStage.AI_COMPLETE, StudyStage.REPORTING))
        n_done = len(completed)

        workflow_st = WorkflowState(
            active_studies=min(1.0, len(self.active_studies) / _MAX_STUDIES),
            completed_studies=min(1.0, n_done / max(1, self.max_studies)),
            processing_stages=[
                n_scan / max(1, _MAX_STUDIES),
                n_ai / max(1, _MAX_STUDIES),
                n_rep / max(1, _MAX_STUDIES),
                n_done / max(1, _MAX_STUDIES),
            ],
            modality_distribution=[
                sum(1 for s in all_studies if s.modality == m) / max(1, len(all_studies))
                for m in [Modality.CT, Modality.MRI, Modality.XR, Modality.US]
            ],
        )

        # Model state
        avail_models = [m for m in self.resource_pool.ai_models if m.is_available]
        n_models = len(self.resource_pool.ai_models)
        model_st = ModelState(
            available_models=len(avail_models) / max(1, n_models),
            mean_suitability=0.7,
            mean_inference_latency=min(
                1.0,
                np.mean([m.base_inference_time for m in self.resource_pool.ai_models]) / 30.0,
            ) if self.resource_pool.ai_models else 0.0,
            mean_gpu_req=np.mean([m.gpu_requirement for m in self.resource_pool.ai_models]) if self.resource_pool.ai_models else 0.0,
            mean_memory_req=np.mean([m.memory_requirement for m in self.resource_pool.ai_models]) if self.resource_pool.ai_models else 0.0,
        )

        # Compute state
        gpu = self.resource_pool.gpu_cluster
        compute_st = ComputeState(
            gpu_utilization=gpu.gpu_utilization,
            cpu_utilization=0.3,  # placeholder
            memory_utilization=gpu.memory_utilization,
            available_compute=1.0 - gpu.gpu_utilization,
        )

        # Human resource state
        rads = self.resource_pool.radiologists
        avail_rads = [r for r in rads if r.is_available]
        workloads = [r.workload for r in rads]
        hr_st = HumanResourceState(
            available_radiologists=len(avail_rads) / max(1, len(rads)),
            mean_workload=float(np.mean(workloads)) if workloads else 0.0,
            reporting_capacity=len(avail_rads) / max(1, len(rads)),
            workload_variance=float(np.var(workloads)) if len(workloads) > 1 else 0.0,
        )

        # SLA state
        total = len(all_studies)
        breached = sum(1 for s in all_studies if s.sla_breached)
        crit_waits = [
            self.clock - s.arrival_time
            for s in waiting
            if s.urgency == Urgency.CRITICAL
        ]
        sla_st = SLAState(
            compliance_rate=1.0 - breached / max(1, total),
            studies_at_risk=min(1.0, at_risk / max(1, total)),
            critical_waiting_time=min(1.0, (max(crit_waits) / 60.0) if crit_waits else 0.0),
        )

        # Clinical context
        clinical_st = ClinicalContext(
            urgency_dist=queue_st.priority_distribution[:],
            modality_mix=workflow_st.modality_distribution[:],
            mean_complexity=np.mean([s.complexity for s in all_studies]) / _MAX_COMPLEXITY if all_studies else 0.5,
            operational_mode=0.0,
        )

        return GlobalState(
            queue=queue_st,
            workflow=workflow_st,
            model=model_st,
            compute=compute_st,
            human_resource=hr_st,
            sla=sla_st,
            clinical=clinical_st,
        )

    # ── Done check ────────────────────────────────────────────────────

    def _check_done(self) -> bool:
        if self.total_steps >= self.max_steps:
            return True
        if (
            not self.waiting_studies
            and not self.active_studies
            and self.event_queue.is_empty()
        ):
            return True
        return False
