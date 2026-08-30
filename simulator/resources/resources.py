from dataclasses import dataclass, field
from typing import Optional, List
from simulator.state.study import Modality

@dataclass
class Scanner:
    scanner_id: str
    modality: Modality
    is_available: bool = True
    current_study_id: Optional[str] = None
    total_scan_time: float = 0.0
    total_scans: int = 0
    
    @property
    def utilization(self) -> float:
        # Assuming normalized utilization for placeholder logic
        # In a real sim, this would compare active time to elapsed time
        return min(1.0, self.total_scan_time / max(1.0, self.total_scan_time))

@dataclass 
class Radiologist:
    radiologist_id: str
    specialties: List[Modality]
    is_available: bool = True
    current_study_id: Optional[str] = None
    completed_reports: int = 0
    total_report_time: float = 0.0
    shift_start: float = 0.0
    shift_end: float = 480.0
    
    @property
    def workload(self) -> float:
        shift_duration = max(1.0, self.shift_end - self.shift_start)
        return min(1.0, self.total_report_time / shift_duration)

@dataclass
class AIModel:
    model_id: str
    name: str
    supported_modalities: List[Modality]
    gpu_requirement: float
    memory_requirement: float
    base_inference_time: float
    is_available: bool = True
    current_study_id: Optional[str] = None
    
    def suitability_score(self, modality: Modality, body_region: str) -> float:
        if modality not in self.supported_modalities:
            return 0.0
        # Dummy logic: models specialized on specific body regions could score higher.
        # Returning a random-ish suitable score [0, 1]
        return 0.85

@dataclass
class GPUCluster:
    total_gpus: int = 4
    total_memory_gb: float = 64.0
    allocated_gpus: float = 0.0
    allocated_memory_gb: float = 0.0
    
    @property
    def gpu_utilization(self) -> float:
        if self.total_gpus == 0: return 0.0
        return self.allocated_gpus / self.total_gpus
    
    @property
    def memory_utilization(self) -> float:
        if self.total_memory_gb == 0.0: return 0.0
        return self.allocated_memory_gb / self.total_memory_gb
    
    def can_allocate(self, gpu_req: float, mem_req: float) -> bool:
        return (self.allocated_gpus + gpu_req <= self.total_gpus) and \
               (self.allocated_memory_gb + mem_req <= self.total_memory_gb)
    
    def allocate(self, gpu_req: float, mem_req: float):
        if self.can_allocate(gpu_req, mem_req):
            self.allocated_gpus += gpu_req
            self.allocated_memory_gb += mem_req
    
    def deallocate(self, gpu_req: float, mem_req: float):
        self.allocated_gpus = max(0.0, self.allocated_gpus - gpu_req)
        self.allocated_memory_gb = max(0.0, self.allocated_memory_gb - mem_req)

@dataclass
class ResourcePool:
    scanners: List[Scanner]
    radiologists: List[Radiologist]
    ai_models: List[AIModel]
    gpu_cluster: GPUCluster
    
    @staticmethod
    def create_default(config: dict) -> 'ResourcePool':
        scanners = []
        scanner_counts = config.get("num_scanners", {"CT": 2, "MRI": 1, "XR": 3, "US": 2})
        for mod_str, count in scanner_counts.items():
            modality = Modality(mod_str)
            for i in range(count):
                scanners.append(Scanner(scanner_id=f"scanner_{mod_str}_{i}", modality=modality))
                
        num_rads = config.get("num_radiologists", 5)
        radiologists = []
        for i in range(num_rads):
            # Give each radiologist a random subset of specialties for now
            rad_specialties = [Modality.CT, Modality.MRI, Modality.XR, Modality.US]
            radiologists.append(Radiologist(radiologist_id=f"rad_{i}", specialties=rad_specialties))
            
        models = [
            AIModel(model_id="model_ct_stroke", name="CT Stroke Detection",
                    supported_modalities=[Modality.CT], gpu_requirement=0.25,
                    memory_requirement=4.0, base_inference_time=5.0),
            AIModel(model_id="model_xr_chest", name="CXR Triage",
                    supported_modalities=[Modality.XR], gpu_requirement=0.15,
                    memory_requirement=2.0, base_inference_time=2.0),
            AIModel(model_id="model_mri_brain", name="MRI Brain Analysis",
                    supported_modalities=[Modality.MRI], gpu_requirement=0.35,
                    memory_requirement=6.0, base_inference_time=8.0),
            AIModel(model_id="model_us_abdomen", name="US Abdomen Screening",
                    supported_modalities=[Modality.US], gpu_requirement=0.20,
                    memory_requirement=3.0, base_inference_time=3.0),
            AIModel(model_id="model_general", name="General Screening",
                    supported_modalities=[Modality.CT, Modality.MRI, Modality.XR, Modality.US],
                    gpu_requirement=0.20, memory_requirement=3.0, base_inference_time=4.0),
        ]
        
        gpu_capacity_fraction = config.get("gpu_capacity_fraction", 1.0)
        gpu_cluster = GPUCluster(
            total_gpus=int(4 * gpu_capacity_fraction),
            total_memory_gb=64.0 * gpu_capacity_fraction
        )
        
        return ResourcePool(scanners=scanners, radiologists=radiologists, ai_models=models, gpu_cluster=gpu_cluster)
