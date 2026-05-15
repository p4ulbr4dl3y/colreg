"""Детектор объектов в инфракрасном диапазоне на основе модели YOLO."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from ultralytics import YOLO

from colreg_vision.core.config import Config


@dataclass
class InfraredDetection:
    """Данные обнаруженного объекта в инфракрасном спектре.

    Атрибуты:
        - bbox: координаты ограничивающей рамки в формате [x1, y1, x2, y2];
        - confidence: уровень уверенности модели при обнаружении;
        - class_id: идентификатор класса объекта;
        - class_name: наименование класса объекта;
        - track_id: идентификатор трека объекта.
    """

    bbox: List[int]
    confidence: float
    class_id: int
    class_name: str
    track_id: int = 0

    @property
    def center_x(self) -> float:
        """Возвращает координату X центра рамки."""
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def center_y(self) -> float:
        """Возвращает координату Y центра рамки."""
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def width(self) -> int:
        """Возвращает ширину рамки объекта в пикселях."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Возвращает высоту рамки объекта в пикселях."""
        return self.bbox[3] - self.bbox[1]


def detect_infrared_objects(
    image: Union[str, Path, np.ndarray],
    config: Optional[Config] = None,
    confidence_threshold: Optional[float] = None,
    model_path: Optional[Union[str, Path]] = None,
    class_filter: Optional[List[int]] = None,
    model: Optional[YOLO] = None,
    use_tracker: bool = False,
) -> List[InfraredDetection]:
    """Выполняет обнаружение объектов на инфракрасном изображении.

    Аргументы:
        - image: входное изображение или путь к нему;
        - config: объект конфигурации системы;
        - confidence_threshold: порог уверенности для фильтрации обнаружений;
        - model_path: путь к весам модели обнаружения;
        - class_filter: список разрешенных идентификаторов классов для фильтрации;
        - model: предобученная модель;
        - use_tracker: флаг использования трекера объектов.

    Возвращает:
        список объектов InfraredDetection с результатами обнаружения.
    """
    if config is None:
        config = Config()
    if model_path is None:
        model_path = config.get_model_path("infrared_detector")
    else:
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = config.base_dir / model_path
    if model is None:
        model = YOLO(str(model_path))
    conf = confidence_threshold or config.infrared_detector.confidence_threshold
    device = config.device
    if use_tracker:
        results = model.track(
            image,
            conf=conf,
            persist=True,
            tracker=config.tracker_type,
            device=device,
            verbose=False,
        )
    else:
        results = model(image, conf=conf, device=device, verbose=False)
    result = results[0]
    detections = []
    if result.boxes is not None:
        boxes = result.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            if class_filter is not None and class_id not in class_filter:
                continue
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            class_name = result.names.get(class_id, f"class_{class_id}")
            track_id = int(boxes.id[i]) if boxes.id is not None else i
            detections.append(
                InfraredDetection(
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                    track_id=track_id,
                )
            )
    return detections
