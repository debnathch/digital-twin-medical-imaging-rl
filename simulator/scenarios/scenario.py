from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ScenarioConfig:
    name: str
    description: str
    arrival_rate_per_hour: float
    urgency_distribution: Dict[str, float]
    modality_distribution: Dict[str, float]
    num_scanners: Dict[str, int]
    num_radiologists: int
    gpu_capacity_fraction: float
    inference_latency_multiplier: float
    radiologist_availability_fraction: float
    simulation_duration_hours: float = 24.0
    studies_per_episode: int = 100

class ScenarioFactory:
    _SCENARIOS = {
        "A": ScenarioConfig(
            name="Scenario A: Normal",
            description="Normal arrival rates and full resources.",
            arrival_rate_per_hour=10.0,
            urgency_distribution={"critical": 0.10, "urgent": 0.25, "routine": 0.65},
            modality_distribution={"CT": 0.30, "MRI": 0.20, "XR": 0.35, "US": 0.15},
            num_scanners={"CT": 2, "MRI": 1, "XR": 3, "US": 2},
            num_radiologists=5,
            gpu_capacity_fraction=1.0,
            inference_latency_multiplier=1.0,
            radiologist_availability_fraction=1.0
        ),
        "B": ScenarioConfig(
            name="Scenario B: Peak",
            description="Peak arrival rate.",
            arrival_rate_per_hour=20.0,
            urgency_distribution={"critical": 0.10, "urgent": 0.25, "routine": 0.65},
            modality_distribution={"CT": 0.30, "MRI": 0.20, "XR": 0.35, "US": 0.15},
            num_scanners={"CT": 2, "MRI": 1, "XR": 3, "US": 2},
            num_radiologists=5,
            gpu_capacity_fraction=1.0,
            inference_latency_multiplier=1.0,
            radiologist_availability_fraction=1.0
        ),
        "C": ScenarioConfig(
            name="Scenario C: High Urgent",
            description="High proportion of urgent cases.",
            arrival_rate_per_hour=10.0,
            urgency_distribution={"critical": 0.40, "urgent": 0.35, "routine": 0.25},
            modality_distribution={"CT": 0.30, "MRI": 0.20, "XR": 0.35, "US": 0.15},
            num_scanners={"CT": 2, "MRI": 1, "XR": 3, "US": 2},
            num_radiologists=5,
            gpu_capacity_fraction=1.0,
            inference_latency_multiplier=1.0,
            radiologist_availability_fraction=1.0
        ),
        "D": ScenarioConfig(
            name="Scenario D: Low Compute",
            description="Half GPU capacity.",
            arrival_rate_per_hour=10.0,
            urgency_distribution={"critical": 0.10, "urgent": 0.25, "routine": 0.65},
            modality_distribution={"CT": 0.30, "MRI": 0.20, "XR": 0.35, "US": 0.15},
            num_scanners={"CT": 2, "MRI": 1, "XR": 3, "US": 2},
            num_radiologists=5,
            gpu_capacity_fraction=0.5,
            inference_latency_multiplier=1.0,
            radiologist_availability_fraction=1.0
        ),
        "E": ScenarioConfig(
            name="Scenario E: Low Radiologist",
            description="Half radiologist availability.",
            arrival_rate_per_hour=10.0,
            urgency_distribution={"critical": 0.10, "urgent": 0.25, "routine": 0.65},
            modality_distribution={"CT": 0.30, "MRI": 0.20, "XR": 0.35, "US": 0.15},
            num_scanners={"CT": 2, "MRI": 1, "XR": 3, "US": 2},
            num_radiologists=5,
            gpu_capacity_fraction=1.0,
            inference_latency_multiplier=1.0,
            radiologist_availability_fraction=0.5
        ),
        "F": ScenarioConfig(
            name="Scenario F: High Latency",
            description="High inference latency (3x).",
            arrival_rate_per_hour=10.0,
            urgency_distribution={"critical": 0.10, "urgent": 0.25, "routine": 0.65},
            modality_distribution={"CT": 0.30, "MRI": 0.20, "XR": 0.35, "US": 0.15},
            num_scanners={"CT": 2, "MRI": 1, "XR": 3, "US": 2},
            num_radiologists=5,
            gpu_capacity_fraction=1.0,
            inference_latency_multiplier=3.0,
            radiologist_availability_fraction=1.0
        )
    }

    _ALIASES = {
        "normal": "A", "peak": "B", "high_urgent": "C",
        "low_compute": "D", "low_radiologist": "E", "high_latency": "F",
    }

    @staticmethod
    def create(name: str) -> ScenarioConfig:
        key = ScenarioFactory._ALIASES.get(name, name)
        if key in ScenarioFactory._SCENARIOS:
            return ScenarioFactory._SCENARIOS[key]
        raise ValueError(f"Unknown scenario name: {name}")

    @staticmethod
    def get_training_scenarios() -> List[ScenarioConfig]:
        return [ScenarioFactory._SCENARIOS[k] for k in ["A", "B", "C"]]

    @staticmethod
    def get_test_scenarios() -> List[ScenarioConfig]:
        return [ScenarioFactory._SCENARIOS[k] for k in ["D", "E", "F"]]

    @staticmethod
    def get_all_scenarios() -> List[ScenarioConfig]:
        return list(ScenarioFactory._SCENARIOS.values())
