"""
Infrared object detection module for night mode.

Detects objects in infrared/thermal images using YOLO.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

from config import Config


@dataclass
class InfraredDetection:
    """Represents a detected object in infrared image."""

    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


def detect_infrared_objects(
    image: Union[str, Path, np.ndarray],
    config: Optional[Config] = None,
    confidence_threshold: Optional[float] = None,
    model_path: Optional[Union[str, Path]] = None,
    class_filter: Optional[List[int]] = None,
) -> List[InfraredDetection]:
    """
    Detect objects in infrared/thermal images.

    This function is pipeline-agnostic - accepts any image and returns
    detections. Works with both day and night infrared imagery.

    Args:
        image: Input image as file path or numpy array (BGR/RGB).
        config: Configuration object. Uses default if None.
        confidence_threshold: Override default confidence threshold.
        model_path: Path to YOLO model weights for infrared detection.
        class_filter: Optional list of class IDs to filter results.

    Returns:
        List of InfraredDetection objects. Empty list if no objects detected.

    Example:
        >>> image = cv2.imread('night_scene.png')
        >>> detections = detect_infrared_objects(image)
        >>> for det in detections:
        ...     print(f"Object: {det.class_name}, conf: {det.confidence:.2f}")
    """
    if config is None:
        config = Config()

    # Resolve model path
    if model_path is None:
        model_path = config.get_model_path("infrared_detector")
    else:
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = config.base_dir / model_path

    # Load image
    if isinstance(image, (str, Path)):
        image_cv = cv2.imread(str(image))
        if image_cv is None:
            raise ValueError(f"Failed to load image: {image}")
        image = image_cv
    elif not isinstance(image, np.ndarray):
        raise TypeError("Image must be a file path or numpy array")

    # Load model and run inference
    model = YOLO(str(model_path))
    results = model(
        image,
        conf=confidence_threshold or config.infrared_detector.confidence_threshold,
    )
    result = results[0]

    detections = []

    if result.boxes is not None:
        boxes = result.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])

            # Apply class filter if specified
            if class_filter is not None and class_id not in class_filter:
                continue

            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            class_name = result.names.get(class_id, f"class_{class_id}")

            detections.append(
                InfraredDetection(
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                )
            )

    return detections
