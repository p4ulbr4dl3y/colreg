"""
Main video analytics pipeline orchestrator.

Coordinates all modules for complete vessel analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

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
    vessel_type: str = "MECH"
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
                    boat_result.vessel_type = "SAIL"
                    boat_result.vessel_type_confidence = (
                        class_result.sailboat_probability * 100
                    )
                else:
                    boat_result.vessel_type = "MECH"
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

            # Lights classification on each boat crop
            for boat in boats:
                lights_statuses = classify_lights(image=boat.crop, config=self.config)
                if lights_statuses:
                    boat.lights_status = lights_statuses[0]

        # Step 4: Day mode - day shapes on each boat crop
        else:
            for boat in boats:
                day_statuses = classify_day_shapes(image=boat.crop, config=self.config)
                if day_statuses:
                    boat.day_shapes_status = day_statuses[0]

        return result

    def process_night(
        self,
        ir_image: Union[str, Path, np.ndarray],
        visible_image: Union[str, Path, np.ndarray],
        boat_confidence: Optional[float] = None,
        classifier_confidence: Optional[float] = None,
        skip_classification: bool = False,
        bbox_offset: Tuple[int, int] = (0, 0),
        bbox_scale: Tuple[float, float, float, float] = (1.0, 1.5, 1.0, 1.0),
        save_debug: bool = False,
        debug_dir: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process night mode with separate IR and visible images.

        Uses IR image for boat detection (thermal signature), then classifies
        navigation lights on visible image crops at the same coordinates.

        Args:
            ir_image: Infrared/thermal image for boat detection.
            visible_image: Visible light image for lights classification.
            boat_confidence: Override boat detection confidence threshold.
            classifier_confidence: Override binary classifier confidence.
            skip_classification: Skip binary classification step.
            bbox_offset: Optional (dx, dy) offset to correct alignment between
                IR and visible sensors. Default (0, 0).
            bbox_scale: Scale factors for bbox expansion (left, top, right, bottom).
                Default (1.0, 1.5, 1.0, 1.0) - expands top by 50% to capture mast/lights.
            save_debug: If True, save intermediate results for debugging.
            debug_dir: Directory for debug images. Uses 'debug/' if None.

        Returns:
            PipelineResult with analysis results.

        Example:
            >>> ir = cv2.imread('thermal_frame.png')
            >>> vis = cv2.imread('visible_frame.png')
            >>> result = pipeline.process_night(ir, vis, bbox_scale=(1.0, 2.0, 1.0, 1.0))
            >>> for boat in result.boats:
            ...     if boat.lights_status:
            ...         print(f"Boat {boat.boat_id}: {boat.lights_status.vessel_type}")
        """
        import os
        from datetime import datetime

        # Load images if paths
        if isinstance(ir_image, (str, Path)):
            ir_cv = cv2.imread(str(ir_image))
            if ir_cv is None:
                raise ValueError(f"Failed to load IR image: {ir_image}")
            ir_image = ir_cv
        elif not isinstance(ir_image, np.ndarray):
            raise TypeError("IR image must be a file path or numpy array")

        if isinstance(visible_image, (str, Path)):
            vis_cv = cv2.imread(str(visible_image))
            if vis_cv is None:
                raise ValueError(f"Failed to load visible image: {visible_image}")
            visible_image = vis_cv
        elif not isinstance(visible_image, np.ndarray):
            raise TypeError("Visible image must be a file path or numpy array")

        # Setup debug directory
        if save_debug:
            debug_dir = debug_dir or "debug"
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_prefix = f"{debug_dir}/night_{timestamp}"

        # Step 1: Detect boats on IR image
        boat_detections = detect_and_crop_boats(
            image=ir_image, config=self.config, confidence_threshold=boat_confidence
        )

        # Save IR detection result
        if save_debug:
            ir_vis = ir_image.copy()
            for det in boat_detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(ir_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(f"{debug_prefix}_ir_detection.png", ir_vis)
            print(f"Saved IR detection: {debug_prefix}_ir_detection.png")

        # Step 2: Binary classification on IR crops, lights on visible crops
        boats = []
        for i, boat_det in enumerate(boat_detections):
            # Original bbox from IR detection
            x1, y1, x2, y2 = boat_det.bbox

            # Calculate center and dimensions
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            # Expand bbox using scale factors
            new_w = w * bbox_scale[0]  # left
            new_h_top = h * bbox_scale[1]  # top (expand upward for mast/lights)
            new_w_right = w * bbox_scale[2]  # right
            new_h_bottom = h * bbox_scale[3]  # bottom

            # Calculate expanded bbox (expand upward and slightly outward)
            exp_x1 = int(max(0, cx - new_w / 2))
            exp_y1 = int(max(0, cy - new_h_top / 2))
            exp_x2 = int(min(ir_image.shape[1], cx + new_w_right / 2))
            exp_y2 = int(min(ir_image.shape[0], cy + new_h_bottom / 2))

            # Adjust for visible image alignment
            adj_bbox = [
                max(0, exp_x1 + bbox_offset[0]),
                max(0, exp_y1 + bbox_offset[1]),
                max(0, exp_x2 + bbox_offset[0]),
                max(0, exp_y2 + bbox_offset[1]),
            ]

            # Crop from IR image for binary classification (use expanded bbox)
            ir_crop = ir_image[exp_y1:exp_y2, exp_x1:exp_x2]

            # Crop from visible image with adjusted bbox for lights classification
            adj_crop = visible_image[adj_bbox[1]:adj_bbox[3], adj_bbox[0]:adj_bbox[2]]

            boat_result = BoatAnalysisResult(
                boat_id=i,
                crop=adj_crop,  # visible crop for lights
                bbox=adj_bbox,
                detection_confidence=boat_det.confidence,
            )

            # Binary classifier on IR crop
            if not skip_classification and ir_crop.size > 0:
                class_result = self.classifier.classify(ir_crop)
                if class_result.is_sailboat:
                    boat_result.vessel_type = "SAIL"
                    boat_result.vessel_type_confidence = (
                        class_result.sailboat_probability * 100
                    )
                else:
                    boat_result.vessel_type = "MECH"
                    boat_result.vessel_type_confidence = (
                        class_result.not_sailboat_probability * 100
                    )

            # Save debug crops
            if save_debug:
                cv2.imwrite(f"{debug_prefix}_boat{i}_ir_crop.png", ir_crop)
                cv2.imwrite(f"{debug_prefix}_boat{i}_vis_crop.png", adj_crop)

            boats.append(boat_result)

        # Initialize result
        result = PipelineResult(image=visible_image, is_night=True, boats=boats)

        # Step 3: Infrared detection on full IR image
        result.infrared_detections = detect_infrared_objects(
            image=ir_image, config=self.config
        )

        # Step 4: Lights classification on visible crops
        if save_debug:
            lights_vis = visible_image.copy()

        for boat in boats:
            if boat.crop.size > 0:
                lights_statuses = classify_lights(image=boat.crop, config=self.config)
                if lights_statuses:
                    boat.lights_status = lights_statuses[0]

                    # Draw lights detection for debug
                    if save_debug:
                        x1, y1, x2, y2 = boat.bbox
                        cv2.rectangle(lights_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(
                            lights_vis,
                            boat.lights_status.vessel_type,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

        if save_debug:
            cv2.imwrite(f"{debug_prefix}_lights_detection.png", lights_vis)
            print(f"Saved lights detection: {debug_prefix}_lights_detection.png")

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


def draw_results(
    image: np.ndarray,
    result: PipelineResult,
    thickness: int = 2,
    font_scale: float = 0.8,
) -> np.ndarray:
    """
    Draw visualization of pipeline results on the original image.

    Draws bounding boxes with vessel types for each detected boat.

    Args:
        image: Original image (BGR).
        result: PipelineResult with analysis results.
        thickness: Box line thickness.
        font_scale: Font scale for labels.

    Returns:
        Image with drawn results.

    Example:
        >>> pipeline = VideoAnalyticsPipeline()
        >>> result = pipeline.process(image, is_night=False)
        >>> vis = draw_results(image, result)
        >>> cv2.imwrite('output.png', vis)
    """
    output = image.copy()

    for boat in result.boats:
        x1, y1, x2, y2 = boat.bbox

        # Get vessel type (with priority from day_shapes/lights)
        label = boat.final_vessel_type

        # Determine color based on type
        if boat.day_shapes_status and boat.day_shapes_status.is_known_signal:
            color = boat.day_shapes_status.color
        elif boat.lights_status and boat.lights_status.is_known_signal:
            color = boat.lights_status.color
        elif label == "SAIL":
            color = (0, 255, 0)  # Green for sail
        else:
            color = (255, 0, 0)  # Blue for mechanical (BGR)

        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Draw label background
        cv2.rectangle(
            output,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            output,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    # Draw infrared detections (night mode)
    if result.is_night and result.infrared_detections:
        for ir_det in result.infrared_detections:
            x1, y1, x2, y2 = ir_det.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 1)  # Cyan

    return output
