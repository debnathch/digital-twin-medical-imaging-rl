from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class Urgency(Enum):
    CRITICAL = 0
    URGENT = 1
    ROUTINE = 2

class Modality(Enum):
    CT = "CT"
    MRI = "MRI"
    XR = "XR"
    US = "US"

class StudyStage(Enum):
    WAITING = "WAITING"
    TRIAGED = "TRIAGED"
    SCANNING = "SCANNING"
    SCAN_COMPLETE = "SCAN_COMPLETE"
    AI_PROCESSING = "AI_PROCESSING"
    AI_COMPLETE = "AI_COMPLETE"
    REPORTING = "REPORTING"
    COMPLETED = "COMPLETED"

@dataclass
class Study:
    study_id: str
    modality: Modality
    body_region: str
    urgency: Urgency
    complexity: int
    arrival_time: float
    sla_deadline: float
    current_stage: StudyStage = StudyStage.WAITING
    assigned_scanner: Optional[str] = None
    assigned_radiologist: Optional[str] = None
    assigned_model: Optional[str] = None
    triage_time: Optional[float] = None
    scan_start_time: Optional[float] = None
    scan_end_time: Optional[float] = None
    ai_start_time: Optional[float] = None
    ai_end_time: Optional[float] = None
    report_start_time: Optional[float] = None
    report_end_time: Optional[float] = None
    completion_time: Optional[float] = None
    sla_breached: bool = False
    
    @property
    def turnaround_time(self) -> Optional[float]:
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return None
    
    @property
    def queue_waiting_time(self) -> Optional[float]:
        if self.scan_start_time is not None:
            return self.scan_start_time - self.arrival_time
        return None
    
    def is_sla_at_risk(self, current_time: float) -> bool:
        time_remaining = self.sla_deadline - current_time
        total_allowed = self.sla_deadline - self.arrival_time
        if total_allowed <= 0:
            return True
        return (time_remaining / total_allowed) <= 0.20

    @staticmethod
    def get_sla_hours(urgency: Urgency) -> float:
        return {Urgency.CRITICAL: 1.0, Urgency.URGENT: 4.0, Urgency.ROUTINE: 24.0}[urgency]
