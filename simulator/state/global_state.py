from dataclasses import dataclass, field
import numpy as np

@dataclass
class QueueState:
    queue_length: float = 0.0
    urgent_queue_length: float = 0.0
    max_wait: float = 0.0
    mean_wait: float = 0.0
    priority_distribution: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    sla_risk: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.queue_length, self.urgent_queue_length, self.max_wait, self.mean_wait] + self.priority_distribution + [self.sla_risk], dtype=np.float32)

@dataclass
class WorkflowState:
    active_studies: float = 0.0
    completed_studies: float = 0.0
    processing_stages: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    modality_distribution: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    
    def to_array(self) -> np.ndarray:
        return np.array([self.active_studies, self.completed_studies] + self.processing_stages + self.modality_distribution, dtype=np.float32)

@dataclass
class ModelState:
    available_models: float = 0.0
    mean_suitability: float = 0.0
    mean_inference_latency: float = 0.0
    mean_gpu_req: float = 0.0
    mean_memory_req: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.available_models, self.mean_suitability, self.mean_inference_latency, self.mean_gpu_req, self.mean_memory_req], dtype=np.float32)

@dataclass
class ComputeState:
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    available_compute: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.gpu_utilization, self.cpu_utilization, self.memory_utilization, self.available_compute], dtype=np.float32)

@dataclass
class HumanResourceState:
    available_radiologists: float = 0.0
    mean_workload: float = 0.0
    reporting_capacity: float = 0.0
    workload_variance: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.available_radiologists, self.mean_workload, self.reporting_capacity, self.workload_variance], dtype=np.float32)

@dataclass
class SLAState:
    compliance_rate: float = 1.0
    studies_at_risk: float = 0.0
    critical_waiting_time: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.compliance_rate, self.studies_at_risk, self.critical_waiting_time], dtype=np.float32)

@dataclass
class ClinicalContext:
    urgency_dist: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    modality_mix: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    mean_complexity: float = 0.0
    operational_mode: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array(self.urgency_dist + self.modality_mix + [self.mean_complexity, self.operational_mode], dtype=np.float32)

@dataclass
class GlobalState:
    queue: QueueState
    workflow: WorkflowState
    model: ModelState
    compute: ComputeState
    human_resource: HumanResourceState
    sla: SLAState
    clinical: ClinicalContext
    
    def to_array(self) -> np.ndarray:
        return np.concatenate([
            self.queue.to_array(),
            self.workflow.to_array(),
            self.model.to_array(),
            self.compute.to_array(),
            self.human_resource.to_array(),
            self.sla.to_array(),
            self.clinical.to_array()
        ])
    
    @property
    def dimension(self) -> int:
        return len(self.to_array())
