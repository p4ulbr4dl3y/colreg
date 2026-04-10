"""
Модуль обнаружения и кадрирования судов.

Обнаруживает суда на изображениях с помощью YOLO и возвращает вырезанные области.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

from colreg_vision.core.config import Config


@dataclass
class BoatDetection:
    """Представляет обнаруженное судно с его вырезом и метаданными."""

    crop: np.ndarray  # Вырезанное изображение судна (BGR)
    bbox: List[int]  # [x1, y1, x2, y2] в координатах исходного изображения
    confidence: float  # Уверенность обнаружения
    crop_id: int  # Уникальный ID для этого обнаружения

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
    model: Optional[YOLO] = None,
    use_tracker: bool = False,
) -> List[BoatDetection]:
    """
    Обнаружить суда на изображении и вернуть вырезанные области.

    Эта функция не зависит от конвейера — она принимает изображение и возвращает
    обнаружения судов с вырезами. Без побочных эффектов (без сохранения файлов).

    Args:
        image: Входное изображение как путь к файлу (str/Path) или numpy массив (BGR).
        config: Объект конфигурации. Использует Config по умолчанию, если None.
        confidence_threshold: Переопределить порог уверенности по умолчанию.
        class_id: ID класса для обнаружения судов (по умолчанию: 8 для COCO boat).
        model_path: Путь к весам модели YOLO. Используется по умолчанию, если None.
        model: Предварительно загруженная модель YOLO (для кэширования в конвейере).
        use_tracker: Использовать ли трекер (только для видеопотока).

    Returns:
        Список объектов BoatDetection, содержащих вырезы и метаданные.
    """
    if config is None:
        config = Config()

    # Разрешить путь к модели
    if model_path is None:
        model_path = config.get_model_path("boat_detector")
    else:
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = config.base_dir / model_path

    # Загрузить модель
    if model is None:
        model = YOLO(str(model_path))

    # Выполнить инференс (с трекером или без)
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
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])

                # Получить ID трека, если доступен
                track_id = int(boxes.id[i]) if boxes.id is not None else i

                # Вырезать область судна
                crop = image[y1:y2, x1:x2].copy()

                detections.append(
                    BoatDetection(
                        crop=crop, bbox=[x1, y1, x2, y2], confidence=conf, crop_id=track_id
                    )
                )

    return detections
