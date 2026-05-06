from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from ultralytics import YOLO

from colreg_vision.core.config import Config


@dataclass
class BoatDetection:
    """
    Данные обнаруженного судна.

    Атрибуты:
    - crop: изображение с вырезанным объектом судна;
    - bbox: координаты ограничивающей рамки в формате [x1, y1, x2, y2];
    - confidence: уровень уверенности модели при обнаружении;
    - crop_id: уникальный идентификатор судна для отслеживания.
    """
    crop: np.ndarray
    bbox: List[int]
    confidence: float
    crop_id: int

    @property
    def width(self) -> int:
        """Возвращает ширину рамки судна в пикселях."""
        return self.crop.shape[1]

    @property
    def height(self) -> int:
        """Возвращает высоту рамки судна в пикселях."""
        return self.crop.shape[0]


def detect_and_crop_boats(
    image: Union[str, Path, np.ndarray],
    config: Optional[Config] = None,
    confidence_threshold: Optional[float] = None,
    class_id: Optional[int] = None,
    model_path: Optional[Union[str, Path]] = None,
    model: Optional[YOLO] = None,
    use_tracker: bool = False,
) -> List[BoatDetection]:
    """
    Выполняет обнаружение судов на изображении и возвращает список обрезанных областей.

    Аргументы:
    - image: входное изображение или путь к нему;
    - config: объект конфигурации системы;
    - confidence_threshold: порог уверенности для фильтрации обнаружений;
    - class_id: идентификатор целевого класса для фильтрации;
    - model_path: путь к весам модели обнаружения;
    - model: предобученная модель;
    - use_tracker: флаг использования трекера объектов.

    Возвращает:
    - список объектов BoatDetection с найденными судами.
    """
    if config is None:
        config = Config()
    if model_path is None:
        model_path = config.get_model_path("boat_detector")
    else:
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = config.base_dir / model_path
    if model is None:
        model = YOLO(str(model_path))
    conf = confidence_threshold or config.boat_detector.confidence_threshold
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
    target_class = class_id if class_id is not None else config.boat_class_id
    if result.boxes is not None:
        boxes = result.boxes
        for i in range(len(boxes)):
            det_class = int(boxes.cls[i])
            if det_class == target_class:
                conf = float(boxes.conf[i])
                (x1, y1, x2, y2) = map(int, boxes.xyxy[i])
                track_id = int(boxes.id[i]) if boxes.id is not None else i
                crop = image[y1:y2, x1:x2].copy()
                detections.append(
                    BoatDetection(
                        crop=crop,
                        bbox=[x1, y1, x2, y2],
                        confidence=conf,
                        crop_id=track_id,
                    )
                )
    return detections
