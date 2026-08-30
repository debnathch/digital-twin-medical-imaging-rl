from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import uuid

class EventType(Enum):
    STUDY_ARRIVAL = "STUDY_ARRIVAL"
    DICOM_RECEIVED = "DICOM_RECEIVED"
    HL7_ADT = "HL7_ADT"
    HL7_ORM = "HL7_ORM"
    HL7_ORU = "HL7_ORU"
    STUDY_TRIAGED = "STUDY_TRIAGED"
    SCAN_STARTED = "SCAN_STARTED"
    SCAN_COMPLETED = "SCAN_COMPLETED"
    AI_INFERENCE_STARTED = "AI_INFERENCE_STARTED"
    AI_INFERENCE_COMPLETED = "AI_INFERENCE_COMPLETED"
    RADIOLOGIST_ASSIGNED = "RADIOLOGIST_ASSIGNED"
    REPORT_STARTED = "REPORT_STARTED"
    REPORT_COMPLETED = "REPORT_COMPLETED"
    SLA_BREACH = "SLA_BREACH"
    RESOURCE_AVAILABLE = "RESOURCE_AVAILABLE"

@dataclass
class Event:
    event_id: str
    event_type: EventType
    timestamp: float
    study_id: Optional[str] = None
    priority: int = 0
    metadata: dict = field(default_factory=dict)
    
    @staticmethod
    def create(event_type: EventType, timestamp: float, study_id: str = None, priority: int = 0, metadata: dict = None) -> 'Event':
        return Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=timestamp,
            study_id=study_id,
            priority=priority,
            metadata=metadata or {}
        )
