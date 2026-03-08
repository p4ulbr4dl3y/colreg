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

from config import Config


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

    Returns:
        Список объектов BoatDetection, содержащих вырезы и метаданные.
        Пустой список, если суда не обнаружены или загрузка изображения не удалась.

    Пример:
        >>> image = cv2.imread('frame.jpg')
        >>> detections = detect_and_crop_boats(image)
        >>> for det in detections:
        ...     print(f"Судно обнаружено: {det.confidence:.2f}, размер: {det.width}x{det.height}")
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

    # Загрузить изображение
    if isinstance(image, (str, Path)):
        image_cv = cv2.imread(str(image))
        if image_cv is None:
            raise ValueError(f"Не удалось загрузить изображение: {image}")
        image = image_cv
    elif not isinstance(image, np.ndarray):
        raise TypeError("Изображение должно быть путём к файлу или numpy массивом")

    # Загрузить модель и выполнить инференс
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

                # Вырезать область судна
                crop = image[y1:y2, x1:x2].copy()

                detections.append(
                    BoatDetection(
                        crop=crop, bbox=[x1, y1, x2, y2], confidence=conf, crop_id=i
                    )
                )

    return detections
