"""
Main video analytics pipeline orchestrator.

Coordinates all modules for complete vessel analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from .binary_classifier import BinaryClassifier, ClassificationResult
    from .boat_detector import BoatDetection, detect_and_crop_boats
    from .config import Config
    from .day_shapes import VesselStatus, classify_day_shapes
    from .infrared_detector import InfraredDetection, detect_infrared_objects
    from .lights import VesselLightStatus, classify_lights
except ImportError:
    from binary_classifier import BinaryClassifier, ClassificationResult
    from boat_detector import BoatDetection, detect_and_crop_boats
    from config import Config
    from day_shapes import VesselStatus, classify_day_shapes
    from infrared_detector import InfraredDetection, detect_infrared_objects
    from lights import VesselLightStatus, classify_lights


@dataclass
class BoatAnalysisResult:
    """Complete analysis result for a single boat."""

    boat_id: int
    crop: np.ndarray
    bbox: List[int]
    detection_confidence: float

    # Binary classification
    is_sailboat: bool = False
    sailboat_confidence: float = 0.0

    # Day shapes (if applicable)
    day_shapes_status: Optional[VesselStatus] = None

    # Lights (if applicable, night mode)
    lights_status: Optional[VesselLightStatus] = None

    # Infrared detections (night mode)
    infrared_detections: List[InfraredDetection] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Complete pipeline result for an image."""

    image: np.ndarray
    is_night: bool

    # All boat analyses
    boats: List[BoatAnalysisResult] = field(default_factory=list)

    # Infrared detections (night mode, full image)
    infrared_detections: List[InfraredDetection] = field(default_factory=list)

    # Day shapes (full image)
    day_shapes_statuses: List[VesselStatus] = field(default_factory=list)

    # Lights (full image, night mode)
    lights_statuses: List[VesselLightStatus] = field(default_factory=list)

    @property
    def boat_count(self) -> int:
        return len(self.boats)

    @property
    def sailboat_count(self) -> int:
        return sum(1 for b in self.boats if b.is_sailboat)


class VideoAnalyticsPipeline:
    """
    Complete video analytics pipeline for maritime navigation.

    This pipeline coordinates all modules:
    1. Boat detection and cropping (YOLO)
    2. Binary classification (sailboat vs not sailboat)
    3. Infrared detection (night mode)
    4. Day-shapes classification (day mode)
    5. Navigation lights classification (night mode)

    All functions are also available individually for standalone use.

    Example:
        >>> pipeline = VideoAnalyticsPipeline()
        >>> image = cv2.imread('frame.jpg')

        >>> # Day mode analysis
        >>> result = pipeline.process(image, is_night=False)
        >>> print(f"Boats: {result.boat_count}, Sailboats: {result.sailboat_count}")

        >>> # Night mode analysis
        >>> result = pipeline.process(image, is_night=True)
        >>> for boat in result.boats:
        ...     if boat.lights_status:
        ...         print(f"Boat {boat.boat_id}: {boat.lights_status.status}")
    """

    def __init__(
        self, config: Optional[Config] = None, classifier_device: Optional[str] = None
    ):
        """
        Initialize the pipeline.

        Args:
            config: Configuration object. Uses default if None.
            classifier_device: Device for binary classifier ('cuda', 'cpu', or None).
        """
        self.config = config or Config()

        # Initialize binary classifier (heavy, load once)
        self._classifier = None
        self._classifier_device = classifier_device

    @property
    def classifier(self) -> BinaryClassifier:
        """Lazy-load binary classifier."""
        if self._classifier is None:
            self._classifier = BinaryClassifier(
                config=self.config, device=self._classifier_device
            )
        return self._classifier

    def process(
        self,
        image: Union[str, Path, np.ndarray],
        is_night: bool = False,
        boat_confidence: Optional[float] = None,
        classifier_confidence: Optional[float] = None,
        skip_classification: bool = False,
    ) -> PipelineResult:
        """
        Process an image through the complete pipeline.

        Args:
            image: Input image as file path or numpy array (BGR).
            is_night: If True, use infrared and lights detection.
            boat_confidence: Override boat detection confidence threshold.
            classifier_confidence: Override binary classifier confidence.
            skip_classification: Skip binary classification step.

        Returns:
            PipelineResult with all analysis results.

        Example:
            >>> image = cv2.imread('frame.jpg')
            >>> result = pipeline.process(image, is_night=False)
            >>> for boat in result.boats:
            ...     print(f"Sailboat: {boat.is_sailboat}, conf: {boat.sailboat_confidence:.1f}%")
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image_cv = cv2.imread(str(image))
            if image_cv is None:
                raise ValueError(f"Failed to load image: {image}")
            image = image_cv
        elif not isinstance(image, np.ndarray):
            raise TypeError("Image must be a file path or numpy array")

        # Step 1: Detect and crop boats
        boat_detections = detect_and_crop_boats(
            image=image, config=self.config, confidence_threshold=boat_confidence
        )

        # Step 2: Binary classification for each boat
        boats = []
        for i, boat_det in enumerate(boat_detections):
            boat_result = BoatAnalysisResult(
                boat_id=i,
                crop=boat_det.crop,
                bbox=boat_det.bbox,
                detection_confidence=boat_det.confidence,
            )

            # Classify as sailboat or not
            if not skip_classification:
                class_result = self.classifier.classify(boat_det.crop)
                boat_result.is_sailboat = class_result.is_sailboat
                boat_result.sailboat_confidence = (
                    class_result.sailboat_probability * 100
                    if class_result.is_sailboat
                    else class_result.not_sailboat_probability * 100
                )

            boats.append(boat_result)

        # Initialize result
        result = PipelineResult(image=image, is_night=is_night, boats=boats)

        # Step 3: Night mode - infrared and lights
        if is_night:
            # Infrared detection on full image
            result.infrared_detections = detect_infrared_objects(
                image=image, config=self.config
            )

            # Lights classification on full image
            result.lights_statuses = classify_lights(image=image, config=self.config)

            # Also classify lights for each boat crop
            for boat in boats:
                lights_statuses = classify_lights(image=boat.crop, config=self.config)
                if lights_statuses:
                    boat.lights_status = lights_statuses[0]

        # Step 4: Day mode - day shapes
        else:
            result.day_shapes_statuses = classify_day_shapes(
                image=image, config=self.config
            )

            # Also classify day shapes for each boat crop
            for boat in boats:
                day_statuses = classify_day_shapes(image=boat.crop, config=self.config)
                if day_statuses:
                    boat.day_shapes_status = day_statuses[0]

        return result

    def process_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        is_night: bool = False,
        **kwargs,
    ) -> List[PipelineResult]:
        """
        Process multiple images.

        Args:
            images: List of images (paths or numpy arrays).
            is_night: Night mode flag.
            **kwargs: Additional arguments passed to process().

        Returns:
            List of PipelineResult objects.
        """
        return [self.process(image, is_night, **kwargs) for image in images]

    def detect_boats_only(
        self, image: Union[str, Path, np.ndarray]
    ) -> List[BoatDetection]:
        """
        Detect boats without full pipeline processing.

        Args:
            image: Input image.

        Returns:
            List of BoatDetection objects.
        """
        return detect_and_crop_boats(image, config=self.config)

    def classify_boat_crop(self, crop: np.ndarray) -> ClassificationResult:
        """
        Classify a single boat crop.

        Args:
            crop: Boat crop image (BGR).

        Returns:
            ClassificationResult.
        """
        return self.classifier.classify(crop)
