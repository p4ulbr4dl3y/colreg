"""
Navigation lights classification module.

Classifies vessel status based on navigation lights.
Implements COLREGS rules for night signals.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from .config import Config
except ImportError:
    from config import Config


@dataclass
class LightDetection:
    """Represents a detected navigation light."""

    class_id: int
    class_name: str  # 'white', 'red', 'green'
    bbox: List[int]  # [x1, y1, x2, y2]
    center_x: float
    center_y: float
    confidence: float


@dataclass
class VesselLightStatus:
    """Classified vessel status from navigation lights."""

    status: str  # e.g., 'NUC', 'RAM', 'CBD', 'Fishing', 'Trawling'
    bbox: List[int]  # Combined bounding box for the group
    color: Tuple[int, int, int]  # BGR color for visualization
    lights: List[LightDetection]  # Constituent lights
    sequence: List[int]  # Class sequence from top to bottom

    @property
    def is_known_signal(self) -> bool:
        return self.status not in ["Unknown", "Неизвестный сигнал"]


# COLREGS navigation lights rules
LIGHTS_RULES = {
    # NUC: Not Under Command - Red, Red
    "NUC": {
        "sequence": [1, 1],
        "color": (0, 255, 0),  # Green
        "description": "Not Under Command",
    },
    # RAM: Restricted Ability to Maneuver - Red, White, Red
    "RAM": {
        "sequence": [1, 0, 1],
        "color": (0, 255, 0),
        "description": "Restricted Ability to Maneuver",
    },
    # Fishing (not trawling) - Red, White
    "Fishing": {
        "sequence": [1, 0],
        "color": (0, 255, 0),
        "description": "Engaged in Fishing",
    },
    # Trawling - Green, White
    "Trawling": {
        "sequence": [2, 0],
        "color": (0, 255, 0),
        "description": "Engaged in Trawling",
    },
    # CBD: Constrained by Draft - Red, Red, Red
    "CBD": {
        "sequence": [1, 1, 1],
        "color": (0, 255, 0),
        "description": "Constrained by Draft",
    },
}


def _group_by_mast(
    detections: List[LightDetection], x_tolerance: int = 40
) -> List[List[LightDetection]]:
    """
    Group detections by mast (vertical alignment).

    Args:
        detections: List of detected lights.
        x_tolerance: Maximum horizontal distance to consider same mast.

    Returns:
        List of groups, each group contains lights on same mast.
    """
    if not detections:
        return []

    # Sort by vertical position (top to bottom)
    sorted_detections = sorted(detections, key=lambda x: x.center_y)

    groups = []
    current_group = [sorted_detections[0]]

    for i in range(1, len(sorted_detections)):
        prev = current_group[-1]
        curr = sorted_detections[i]

        # Check if on same mast (similar X position)
        if abs(curr.center_x - prev.center_x) < x_tolerance:
            current_group.append(curr)
        else:
            # New mast
            groups.append(current_group)
            current_group = [curr]

    groups.append(current_group)
    return groups


def _classify_group(
    group: List[LightDetection], rules: dict = LIGHTS_RULES
) -> VesselLightStatus:
    """
    Classify vessel status based on light sequence.

    Args:
        group: List of lights on same mast.
        rules: Dictionary of COLREGS rules.

    Returns:
        VesselLightStatus with classification result.
    """
    # Get sequence (top to bottom)
    sequence = [d.class_id for d in group]

    # Match against rules
    status_name = "Unknown"
    color = (0, 0, 255)  # Red (unknown)

    for name, rule in rules.items():
        if sequence == rule["sequence"]:
            status_name = name
            color = rule["color"]
            break

    # Calculate combined bounding box
    x1_min = min(d.bbox[0] for d in group)
    y1_min = min(d.bbox[1] for d in group)
    x2_max = max(d.bbox[2] for d in group)
    y2_max = max(d.bbox[3] for d in group)

    return VesselLightStatus(
        status=status_name,
        bbox=[x1_min, y1_min, x2_max, y2_max],
        color=color,
        lights=group,
        sequence=sequence,
    )


def classify_lights(
    image: Union[str, Path, np.ndarray],
    config: Optional[Config] = None,
    confidence_threshold: Optional[float] = None,
    model_path: Optional[Union[str, Path]] = None,
    x_tolerance: Optional[int] = None,
    return_detections: bool = False,
) -> Union[
    List[VesselLightStatus], Tuple[List[VesselLightStatus], List[LightDetection]]
]:
    """
    Classify vessel status based on navigation lights.

    This function is pipeline-agnostic - accepts any image and returns
    classified vessel statuses based on COLREGS navigation light signals.

    Args:
        image: Input image as file path or numpy array (BGR).
        config: Configuration object. Uses default if None.
        confidence_threshold: Override default confidence threshold.
        model_path: Path to YOLO model weights.
        x_tolerance: Horizontal tolerance for mast grouping (pixels).
        return_detections: If True, also return raw detections.

    Returns:
        List of VesselLightStatus objects. If return_detections=True,
        also returns list of raw LightDetection objects.

    Example:
        >>> image = cv2.imread('night_vessel.png')
        >>> statuses = classify_lights(image)
        >>> for status in statuses:
        ...     print(f"Vessel status: {status.status}")
    """
    if config is None:
        config = Config()

    # Resolve model path
    if model_path is None:
        model_path = config.get_model_path("lights")
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
        image, conf=confidence_threshold or config.lights.confidence_threshold
    )
    result = results[0]

    # Extract detections
    detections = []
    if result.boxes is not None:
        boxes = result.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])

            detections.append(
                LightDetection(
                    class_id=class_id,
                    class_name=config.lights_classes.get(class_id, f"class_{class_id}"),
                    bbox=[x1, y1, x2, y2],
                    center_x=(x1 + x2) / 2,
                    center_y=(y1 + y2) / 2,
                    confidence=conf,
                )
            )

    # Group by mast and classify
    groups = _group_by_mast(detections, x_tolerance or config.grouping_x_tolerance)
    statuses = [_classify_group(group) for group in groups]

    if return_detections:
        return statuses, detections
    return statuses


def draw_lights_results(
    image: np.ndarray,
    statuses: List[VesselLightStatus],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw classification results on image.

    Args:
        image: Input image (BGR).
        statuses: List of VesselLightStatus objects.
        thickness: Box line thickness.
        font_scale: Font scale for labels.

    Returns:
        Image with drawn results.
    """
    output = image.copy()

    for status in statuses:
        x1, y1, x2, y2 = status.bbox

        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), status.color, thickness)

        # Draw label
        label = f"{status.status}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Label background
        cv2.rectangle(
            output,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            status.color,
            -1,
        )

        # Label text
        cv2.putText(
            output,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    return output
