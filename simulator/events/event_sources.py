import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np

from simulator.events.event import Event, EventType

class EventSource(ABC):
    @abstractmethod
    def generate_events(self, rng: np.random.Generator, start_time: float, end_time: float, config: Any) -> List[Event]:
        pass

class MockEventSource(EventSource):
    def generate_events(self, rng: np.random.Generator, start_time: float, end_time: float, config: Any) -> List[Event]:
        events = []
        current_time = start_time
        
        # arrival_rate_per_hour = config.arrival_rate_per_hour
        arrival_rate_per_hour = getattr(config, 'arrival_rate_per_hour', 10.0)
        arrival_rate_per_minute = arrival_rate_per_hour / 60.0
        
        urgencies = ["CRITICAL", "URGENT", "ROUTINE"]
        urgency_probs = [0.10, 0.25, 0.65]
        if hasattr(config, 'urgency_distribution'):
            urgency_probs = [
                config.urgency_distribution.get("critical", 0.10),
                config.urgency_distribution.get("urgent", 0.25),
                config.urgency_distribution.get("routine", 0.65)
            ]
            total_u = sum(urgency_probs)
            urgency_probs = [p / total_u for p in urgency_probs]
            
        modalities = ["CT", "MRI", "XR", "US"]
        modality_probs = [0.30, 0.20, 0.35, 0.15]
        if hasattr(config, 'modality_distribution'):
            modality_probs = [
                config.modality_distribution.get("CT", 0.30),
                config.modality_distribution.get("MRI", 0.20),
                config.modality_distribution.get("XR", 0.35),
                config.modality_distribution.get("US", 0.15)
            ]
            total_m = sum(modality_probs)
            modality_probs = [p / total_m for p in modality_probs]
            
        body_regions = ["head", "chest", "abdomen", "spine", "extremity", "pelvis"]
        
        while current_time < end_time:
            # Generate next arrival time
            inter_arrival = rng.exponential(1.0 / arrival_rate_per_minute)
            current_time += inter_arrival
            
            if current_time >= end_time:
                break
                
            study_id = str(uuid.uuid4())
            modality = rng.choice(modalities, p=modality_probs)
            urgency = rng.choice(urgencies, p=urgency_probs)
            body_region = rng.choice(body_regions)
            complexity = int(rng.integers(1, 6))
            
            metadata = {
                "StudyInstanceUID": f"1.2.840.{rng.integers(100000, 999999)}.{uuid.uuid4().hex[:8]}",
                "modality": modality,
                "body_region": body_region,
                "urgency": urgency,
                "complexity": complexity,
                "acquisition_timestamp": current_time
            }
            
            # Priority: Critical=0, Urgent=1, Routine=2
            priority = 0 if urgency == "CRITICAL" else (1 if urgency == "URGENT" else 2)
            
            events.append(Event.create(
                event_type=EventType.STUDY_ARRIVAL,
                timestamp=current_time,
                study_id=study_id,
                priority=priority,
                metadata=metadata
            ))
            
        return events

class DICOMEventSource(EventSource):
    def generate_events(self, rng: np.random.Generator, start_time: float, end_time: float, config: Any) -> List[Event]:
        # Stub implementation
        return []

class HL7EventSource(EventSource):
    def generate_events(self, rng: np.random.Generator, start_time: float, end_time: float, config: Any) -> List[Event]:
        # Stub implementation
        return []

class ReplayEventSource(EventSource):
    def __init__(self, filepath: str):
        self.filepath = filepath
        
    def generate_events(self, rng: np.random.Generator, start_time: float, end_time: float, config: Any) -> List[Event]:
        events = []
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                for item in data:
                    timestamp = float(item.get("timestamp", start_time))
                    if start_time <= timestamp < end_time:
                        events.append(Event(
                            event_id=item.get("event_id", str(uuid.uuid4())),
                            event_type=EventType(item.get("event_type", "STUDY_ARRIVAL")),
                            timestamp=timestamp,
                            study_id=item.get("study_id"),
                            priority=item.get("priority", 0),
                            metadata=item.get("metadata", {})
                        ))
        except Exception:
            pass
        return events
