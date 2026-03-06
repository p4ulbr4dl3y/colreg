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
    from .day_shapes import VesselTypeResult, classify_day_shapes
    from .infrared_detector import InfraredDetection, detect_infrared_objects
    from .lights import VesselTypeResult as LightsVesselTypeResult, classify_lights
except ImportError:
    from binary_classifier import BinaryClassifier, ClassificationResult
    from boat_detector import BoatDetection, detect_and_crop_boats
    from config import Config
    from day_shapes import VesselTypeResult, classify_day_shapes
    from infrared_detector import InfraredDetection, detect_infrared_objects
    from lights import VesselTypeResult as LightsVesselTypeResult, classify_lights


@dataclass
class BoatAnalysisResult:
    """Complete analysis result for a single boat."""

    boat_id: int
    crop: np.ndarray
    bbox: List[int]
    detection_confidence: float

    # Binary classification → vessel type (COLREGS 72)
    # 'sailboat' → VesselType.SAIL, 'not_sailboat' → VesselType.MECHANICAL
    vessel_type: str = "Судно с механическим двигателем"
    vessel_type_confidence: float = 0.0

    # Day shapes (if applicable) - overrides vessel type if detected
    day_shapes_status: Optional[VesselTypeResult] = None

    # Lights (if applicable, night mode) - overrides vessel type if detected
    lights_status: Optional[LightsVesselTypeResult] = None

    # Infrared detections (night mode)
    infrared_detections: List[InfraredDetection] = field(default_factory=list)

    @property
    def final_vessel_type(self) -> str:
        """
        Get final vessel type according to COLREGS 72 priority.

        Priority (highest to lowest):
        1. NUC (Not Under Command)
        2. RAM (Restricted Ability to Maneuver)
        3. CBD (Constrained by Draft)
        4. Fishing/Trawling
        5. Sail / Mechanical (from binary classifier)
        """
        # Day shapes have highest priority
        if self.day_shapes_status and self.day_shapes_status.is_known_signal:
            return self.day_shapes_status.vessel_type

        # Lights classification for night mode
        if self.lights_status and self.lights_status.is_known_signal:
            return self.lights_status.vessel_type

        # Default from binary classifier
        return self.vessel_type


@dataclass
class PipelineResult:
    """Complete pipeline result for an image."""

    image: np.ndarray
    is_night: bool

    # All boat analyses
    boats: List[BoatAnalysisResult] = field(default_factory=list)

    # Infrared detections (night mode, full image)
    infrared_detections: List[InfraredDetection] = field(default_factory=list)

    # Day shapes (full image) - vessel types from day shapes
    day_shapes_statuses: List[VesselTypeResult] = field(default_factory=list)

    # Lights (full image, night mode) - vessel types from lights
    lights_statuses: List[LightsVesselTypeResult] = field(default_factory=list)

    @property
    def boat_count(self) -> int:
        return len(self.boats)

    @property
    def sailboat_count(self) -> int:
        """Count sailboats (for backward compatibility)."""
        return sum(1 for b in self.boats if b.vessel_type == "Парусное судно")

    @property
    def mechanical_count(self) -> int:
        """Count mechanical vessels (default type)."""
        return sum(1 for b in self.boats if b.final_vessel_type == "Судно с механическим двигателем")


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

        # Step 2: Binary classification for each boat → vessel type
        boats = []
        for i, boat_det in enumerate(boat_detections):
            boat_result = BoatAnalysisResult(
                boat_id=i,
                crop=boat_det.crop,
                bbox=boat_det.bbox,
                detection_confidence=boat_det.confidence,
            )

            # Binary classifier: sailboat vs not_sailboat → vessel type
            if not skip_classification:
                class_result = self.classifier.classify(boat_det.crop)
                if class_result.is_sailboat:
                    boat_result.vessel_type = "Парусное судно"
                    boat_result.vessel_type_confidence = (
                        class_result.sailboat_probability * 100
                    )
                else:
                    boat_result.vessel_type = "Судно с механическим двигателем"
                    boat_result.vessel_type_confidence = (
                        class_result.not_sailboat_probability * 100
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
