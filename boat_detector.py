"""
Boat detection and cropping module.

Detects boats in images using YOLO and returns cropped regions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

from config import Config


@dataclass
class BoatDetection:
    """Represents a detected boat with its crop and metadata."""

    crop: np.ndarray  # The cropped boat image (BGR)
    bbox: List[int]  # [x1, y1, x2, y2] in original image coordinates
    confidence: float  # Detection confidence
    crop_id: int  # Unique ID for this detection

    @property
    def width(self) -> int:
        return self.crop.shape[1]

    @property
    def height(self) -> int:
        return self.crop.shape[0]


def detect_and_crop_boats(
    image: Union[str, Path, np.ndarray],
    config: Optional[Config] = None,
    confidence_threshold: Optional[float] = None,
    class_id: Optional[int] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> List[BoatDetection]:
    """
    Detect boats in an image and return cropped regions.

    This function is pipeline-agnostic - it accepts an image and returns
    boat detections with crops. No side effects (no file saving).

    Args:
        image: Input image as file path (str/Path) or numpy array (BGR).
        config: Configuration object. Uses default Config if None.
        confidence_threshold: Override default confidence threshold.
        class_id: Class ID for boat detection (default: 8 for COCO boat).
        model_path: Path to YOLO model weights. Uses default if None.

    Returns:
        List of BoatDetection objects containing crops and metadata.
        Empty list if no boats detected or image loading fails.

    Example:
        >>> image = cv2.imread('frame.jpg')
        >>> detections = detect_and_crop_boats(image)
        >>> for det in detections:
        ...     print(f"Boat detected: {det.confidence:.2f}, size: {det.width}x{det.height}")
    """
    if config is None:
        config = Config()

    # Resolve model path
    if model_path is None:
        model_path = config.get_model_path("boat_detector")
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
        image, conf=confidence_threshold or config.boat_detector.confidence_threshold
    )
    result = results[0]

    detections = []
    target_class = class_id if class_id is not None else config.boat_class_id

    if result.boxes is not None:
        boxes = result.boxes
        for i in range(len(boxes)):
            det_class = int(boxes.cls[i])

            if det_class == target_class:
                conf = float(boxes.conf[i])
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])

                # Crop the boat region
                crop = image[y1:y2, x1:x2].copy()

                detections.append(
                    BoatDetection(
                        crop=crop, bbox=[x1, y1, x2, y2], confidence=conf, crop_id=i
                    )
                )

    return detections
