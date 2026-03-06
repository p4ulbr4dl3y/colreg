"""
Configuration module for the video analytics pipeline.

Centralized configuration for model paths, class mappings, and thresholds.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class ModelConfig:
    """Model configuration with paths and parameters."""

    path: str
    confidence_threshold: float = 0.5


@dataclass
class Config:
    """
    Centralized configuration for the video analytics pipeline.

    All paths are relative to the video-analytics directory.
    """

    # Base directory (video-analytics)
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)

    # ==================== MODEL PATHS ====================

    # Boat detection (YOLO)
    boat_detector: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            path="models/yolo11n.pt", confidence_threshold=0.5
        )
    )

    # Binary classifier (sailboat vs not sailboat)
    binary_classifier: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            path="models/binary-classifier.pth", confidence_threshold=0.5
        )
    )

    # Infrared detection (night mode)
    infrared_detector: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            path="models/infrared.pt", confidence_threshold=0.25
        )
    )

    # Day-shapes classification
    day_shapes: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            path="models/day-shapes.pt", confidence_threshold=0.5
        )
    )

    # Lights classification
    lights: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            path="models/lights.pt", confidence_threshold=0.5
        )
    )

    # ==================== CLASS MAPPINGS ====================

    # Day-shapes class mapping
    day_shapes_classes: Dict[int, str] = field(
        default_factory=lambda: {
            0: "ball",
            1: "cone_up",
            2: "cone_down",
            3: "diamond",
            4: "cylinder",
        }
    )

    # Lights class mapping
    lights_classes: Dict[int, str] = field(
        default_factory=lambda: {0: "white", 1: "red", 2: "green"}
    )

    # Binary classifier class names (loaded from checkpoint)
    binary_classifier_classes: List[str] = field(
        default_factory=lambda: ["not_sailboat", "sailboat"]
    )

    # ==================== DETECTION PARAMETERS ====================

    # Boat class ID in COCO dataset
    boat_class_id: int = 8

    # Grouping tolerance for day-shapes and lights (pixels)
    grouping_x_tolerance: int = 40

    # Image size for binary classifier
    classifier_image_size: int = 224

    # Normalization for binary classifier
    classifier_normalize_mean: List[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406]
    )
    classifier_normalize_std: List[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225]
    )

    def get_model_path(self, model_name: str) -> Path:
        """Get absolute path for a model."""
        model_config = getattr(self, model_name)
        if isinstance(model_config, ModelConfig):
            return self.base_dir / model_config.path
        raise ValueError(f"Unknown model: {model_name}")
