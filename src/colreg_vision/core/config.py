from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    path: str
    confidence_threshold: float = 0.5

@dataclass
class Config:
    base_dir: Path = field(default_factory=lambda : Path(__file__).resolve().parent.parent.parent.parent)
    boat_detector: ModelConfig = field(default_factory=lambda : ModelConfig(path='models/boat_detector.pt', confidence_threshold=0.5))
    binary_classifier: ModelConfig = field(default_factory=lambda : ModelConfig(path='models/binary_classifier.pth', confidence_threshold=0.6))
    infrared_detector: ModelConfig = field(default_factory=lambda : ModelConfig(path='models/infrared_detector.pt', confidence_threshold=0.25))
    day_shapes: ModelConfig = field(default_factory=lambda : ModelConfig(path='models/day_shapes.pt', confidence_threshold=0.88))
    lights: ModelConfig = field(default_factory=lambda : ModelConfig(path='models/lights.pt', confidence_threshold=0.6))
    day_shapes_classes: Dict[int, str] = field(default_factory=lambda : {0: 'ball', 1: 'cone_up', 2: 'cone_down', 3: 'diamond', 4: 'cylinder'})
    lights_classes: Dict[int, str] = field(default_factory=lambda : {0: 'white', 1: 'red', 2: 'green'})
    binary_classifier_classes: List[str] = field(default_factory=lambda : ['not_sailboat', 'sailboat'])
    boat_class_id: int = 8
    grouping_x_tolerance: int = 40
    classifier_image_size: int = 224
    classifier_normalize_mean: List[float] = field(default_factory=lambda : [0.485, 0.456, 0.406])
    classifier_normalize_std: List[float] = field(default_factory=lambda : [0.229, 0.224, 0.225])
    device: Optional[str] = None
    use_tracker: bool = True
    tracker_type: str = 'botsort.yaml'

    def get_model_path(self, model_name: str) -> Path:
        model_config = getattr(self, model_name)
        if isinstance(model_config, ModelConfig):
            return self.base_dir / model_config.path
        raise ValueError(f'Unknown model: {model_name}')