from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from colreg_vision.classifiers.binary import BinaryClassifier
from colreg_vision.classifiers.day_shapes import classify_day_shapes
from colreg_vision.classifiers.lights import classify_lights
from colreg_vision.core.config import Config
from colreg_vision.core.types import (
    CLASS_COLORS,
    BoatAnalysisResult,
    PipelineResult,
    SignalResult,
    VesselType,
)
from colreg_vision.detectors.boat import BoatDetection, detect_and_crop_boats
from colreg_vision.detectors.infrared import InfraredDetection, detect_infrared_objects


def expand_bbox(
    bbox: List[int],
    image_shape: Tuple[int, ...],
    bbox_scale: Tuple[float, float, float, float],
) -> List[int]:
    (x1, y1, x2, y2) = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    new_w_left = w * bbox_scale[0]
    new_h_top = h * bbox_scale[1]
    new_w_right = w * bbox_scale[2]
    new_h_bottom = h * bbox_scale[3]
    exp_x1 = int(max(0, cx - new_w_left / 2))
    exp_y1 = int(max(0, cy - new_h_top / 2))
    exp_x2 = int(min(image_shape[1], cx + new_w_right / 2))
    exp_y2 = int(min(image_shape[0], cy + new_h_bottom / 2))
    return [exp_x1, exp_y1, exp_x2, exp_y2]


class VideoAnalyticsPipeline:
    def __init__(
        self, config: Optional[Config] = None, classifier_device: Optional[str] = None
    ):
        self.config = config or Config()
        self._classifier = None
        self._classifier_device = classifier_device
        self._boat_detector = None
        self._infrared_detector = None
        self._day_shapes_model = None
        self._lights_model = None

    @property
    def classifier(self) -> BinaryClassifier:
        if self._classifier is None:
            self._classifier = BinaryClassifier(
                config=self.config, device=self._classifier_device
            )
        return self._classifier

    @property
    def boat_detector(self) -> YOLO:
        if self._boat_detector is None:
            self._boat_detector = YOLO(str(self.config.get_model_path("boat_detector")))
        return self._boat_detector

    @property
    def infrared_detector(self) -> YOLO:
        if self._infrared_detector is None:
            self._infrared_detector = YOLO(
                str(self.config.get_model_path("infrared_detector"))
            )
        return self._infrared_detector

    @property
    def day_shapes_model(self) -> YOLO:
        if self._day_shapes_model is None:
            self._day_shapes_model = YOLO(str(self.config.get_model_path("day_shapes")))
        return self._day_shapes_model

    @property
    def lights_model(self) -> YOLO:
        if self._lights_model is None:
            self._lights_model = YOLO(str(self.config.get_model_path("lights")))
        return self._lights_model

    def process(
        self,
        image: Union[str, Path, np.ndarray],
        is_night: bool = False,
        boat_confidence: Optional[float] = None,
        classifier_confidence: Optional[float] = None,
        skip_classification: bool = False,
        bbox_scale: Tuple[float, float, float, float] = (1.0, 5.0, 1.0, 1.0),
    ) -> PipelineResult:
        if isinstance(image, (str, Path)):
            image_cv = cv2.imread(str(image))
            if image_cv is None:
                raise ValueError(f"Не удалось загрузить изображение: {image}")
            image = image_cv
        elif not isinstance(image, np.ndarray):
            raise TypeError("Изображение должно быть путём к файлу или numpy массивом")
        boat_detections = detect_and_crop_boats(
            image=image,
            config=self.config,
            confidence_threshold=boat_confidence,
            model=self.boat_detector,
            use_tracker=self.config.use_tracker,
        )
        boats = []
        crops_for_classification = []
        for boat_det in boat_detections:
            exp_bbox = expand_bbox(boat_det.bbox, image.shape, bbox_scale)
            (exp_x1, exp_y1, exp_x2, exp_y2) = exp_bbox
            crop = image[exp_y1:exp_y2, exp_x1:exp_x2]
            boat_result = BoatAnalysisResult(
                boat_id=boat_det.crop_id,
                crop=crop,
                bbox=exp_bbox,
                detection_confidence=boat_det.confidence,
            )
            boats.append(boat_result)
            crops_for_classification.append(boat_det.crop)
        if not skip_classification and crops_for_classification:
            class_results = self.classifier.classify_batch(crops_for_classification)
            for boat, class_result in zip(boats, class_results):
                if class_result.is_sailboat:
                    boat.vessel_type = "SAIL"
                    boat.vessel_type_confidence = (
                        class_result.sailboat_probability * 100
                    )
                else:
                    boat.vessel_type = "MECH"
                    boat.vessel_type_confidence = (
                        class_result.not_sailboat_probability * 100
                    )
        result = PipelineResult(image=image, is_night=is_night, boats=boats)
        if is_night:
            result.infrared_detections = detect_infrared_objects(
                image=image,
                config=self.config,
                model=self.infrared_detector,
                use_tracker=self.config.use_tracker,
            )
            for boat in boats:
                if boat.crop.size == 0:
                    continue
                lights_statuses = classify_lights(
                    image=boat.crop, config=self.config, model=self.lights_model
                )
                valid_statuses = []
                for status in lights_statuses:
                    if not status.is_known_signal:
                        continue
                    signal_w = status.bbox[2] - status.bbox[0]
                    signal_h = status.bbox[3] - status.bbox[1]
                    signal_area = signal_w * signal_h
                    (crop_h, crop_w) = boat.crop.shape[:2]
                    crop_area = crop_h * crop_w
                    if signal_area < crop_area * 0.3 and signal_w < crop_w * 0.6:
                        valid_statuses.append(status)
                if valid_statuses:
                    boat.lights_status = valid_statuses[0]
        else:
            for boat in boats:
                day_statuses = classify_day_shapes(
                    image=boat.crop, config=self.config, model=self.day_shapes_model
                )
                valid_statuses = []
                for status in day_statuses:
                    if not status.is_known_signal:
                        continue
                    signal_w = status.bbox[2] - status.bbox[0]
                    signal_h = status.bbox[3] - status.bbox[1]
                    signal_area = signal_w * signal_h
                    (crop_h, crop_w) = boat.crop.shape[:2]
                    crop_area = crop_h * crop_w
                    if signal_area < crop_area * 0.3 and signal_w < crop_w * 0.6:
                        valid_statuses.append(status)
                if valid_statuses:
                    boat.day_shapes_status = valid_statuses[0]
        return result

    def process_night(
        self,
        ir_image: Union[str, Path, np.ndarray],
        visible_image: Union[str, Path, np.ndarray],
        boat_confidence: Optional[float] = None,
        classifier_confidence: Optional[float] = None,
        skip_classification: bool = False,
        bbox_offset: Tuple[int, int] = (0, 0),
        bbox_scale: Tuple[float, float, float, float] = (1.0, 5.0, 1.0, 1.0),
    ) -> PipelineResult:
        if isinstance(ir_image, (str, Path)):
            ir_cv = cv2.imread(str(ir_image))
            if ir_cv is None:
                raise ValueError(f"Не удалось загрузить ИК-изображение: {ir_image}")
            ir_image = ir_cv
        elif not isinstance(ir_image, np.ndarray):
            raise TypeError(
                "ИК-изображение должно быть путём к файлу или numpy массивом"
            )
        if isinstance(visible_image, (str, Path)):
            vis_cv = cv2.imread(str(visible_image))
            if vis_cv is None:
                raise ValueError(
                    f"Не удалось загрузить видимое изображение: {visible_image}"
                )
            visible_image = vis_cv
        elif not isinstance(visible_image, np.ndarray):
            raise TypeError(
                "Видимое изображение должно быть путём к файлу или numpy массивом"
            )
        ir_detections_raw = detect_infrared_objects(
            image=ir_image,
            config=self.config,
            confidence_threshold=boat_confidence,
            model=self.infrared_detector,
            use_tracker=self.config.use_tracker,
        )
        boat_detections = []
        for ir_det in ir_detections_raw:
            (x1, y1, x2, y2) = ir_det.bbox
            crop = ir_image[y1:y2, x1:x2].copy()
            boat_detections.append(
                BoatDetection(
                    crop=crop,
                    bbox=ir_det.bbox,
                    confidence=ir_det.confidence,
                    crop_id=ir_det.track_id,
                )
            )
        boats = []
        crops_for_classification = []
        for boat_det in boat_detections:
            exp_bbox = expand_bbox(boat_det.bbox, ir_image.shape, bbox_scale)
            (exp_x1, exp_y1, exp_x2, exp_y2) = exp_bbox
            adj_bbox = [
                max(0, exp_x1 + bbox_offset[0]),
                max(0, exp_y1 + bbox_offset[1]),
                max(0, exp_x2 + bbox_offset[0]),
                max(0, exp_y2 + bbox_offset[1]),
            ]
            ir_crop = ir_image[exp_y1:exp_y2, exp_x1:exp_x2]
            adj_crop = visible_image[
                adj_bbox[1] : adj_bbox[3], adj_bbox[0] : adj_bbox[2]
            ]
            boat_result = BoatAnalysisResult(
                boat_id=boat_det.crop_id,
                crop=adj_crop,
                bbox=exp_bbox,
                detection_confidence=boat_det.confidence,
            )
            boats.append(boat_result)
            crops_for_classification.append(ir_crop)
        if not skip_classification and crops_for_classification:
            valid_indices = [
                j for (j, c) in enumerate(crops_for_classification) if c.size > 0
            ]
            if valid_indices:
                valid_crops = [crops_for_classification[j] for j in valid_indices]
                class_results = self.classifier.classify_batch(valid_crops)
                for idx, class_result in zip(valid_indices, class_results):
                    boat = boats[idx]
                    if class_result.is_sailboat:
                        boat.vessel_type = "SAIL"
                        boat.vessel_type_confidence = (
                            class_result.sailboat_probability * 100
                        )
                    else:
                        boat.vessel_type = "MECH"
                        boat.vessel_type_confidence = (
                            class_result.not_sailboat_probability * 100
                        )
        result = PipelineResult(image=ir_image, is_night=True, boats=boats)
        result.infrared_detections = ir_detections_raw
        for boat in boats:
            if boat.crop.size > 0:
                lights_statuses = classify_lights(
                    image=boat.crop, config=self.config, model=self.lights_model
                )
                valid_statuses = []
                for status in lights_statuses:
                    if not status.is_known_signal:
                        continue
                    signal_w = status.bbox[2] - status.bbox[0]
                    signal_h = status.bbox[3] - status.bbox[1]
                    signal_area = signal_w * signal_h
                    (crop_h, crop_w) = boat.crop.shape[:2]
                    crop_area = crop_h * crop_w
                    if signal_area < crop_area * 0.3 and signal_w < crop_w * 0.6:
                        valid_statuses.append(status)
                if valid_statuses:
                    boat.lights_status = valid_statuses[0]
        return result


def draw_results(
    image: np.ndarray,
    result: PipelineResult,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    output = image.copy()
    for boat in result.boats:
        (x1, y1, x2, y2) = boat.bbox
        label = boat.final_vessel_type
        color = CLASS_COLORS.get(label, (255, 255, 255))
        if boat.day_shapes_status and boat.day_shapes_status.is_known_signal:
            color = boat.day_shapes_status.color
        elif boat.lights_status and boat.lights_status.is_known_signal:
            color = boat.lights_status.color
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        confidence_str = f"{boat.final_vessel_type_confidence:.0f}%"
        full_label = f"{label} #{boat.boat_id} ({confidence_str})"
        ((label_w, label_h), baseline) = cv2.getTextSize(
            full_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        text_x = x1 + thickness + 2
        text_y = y1 + label_h + thickness + 2
        if text_y > y2:
            text_y = y2 - baseline
        cv2.rectangle(
            output,
            (x1, y1),
            (
                x1 + label_w + thickness * 2 + 4,
                y1 + label_h + baseline + thickness * 2 + 4,
            ),
            color,
            -1,
        )
        cv2.putText(
            output,
            full_label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255) if sum(color) < 400 else (0, 0, 0),
            thickness,
        )
    return output
