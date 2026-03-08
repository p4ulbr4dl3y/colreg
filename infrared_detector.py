"""
Модуль инфракрасного обнаружения объектов для ночного режима.

Обнаруживает объекты на инфракрасных/тепловых изображениях с помощью YOLO.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

from config import Config


@dataclass
class InfraredDetection:
    """Представляет обнаруженный объект на инфракрасном изображении."""

    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


def detect_infrared_objects(
    image: Union[str, Path, np.ndarray],
    config: Optional[Config] = None,
    confidence_threshold: Optional[float] = None,
    model_path: Optional[Union[str, Path]] = None,
    class_filter: Optional[List[int]] = None,
) -> List[InfraredDetection]:
    """
    Обнаружить объекты на инфракрасных/тепловых изображениях.

    Эта функция не зависит от конвейера — принимает любое изображение и возвращает
    обнаружения. Работает как с дневными, так и с ночными инфракрасными изображениями.

    Args:
        image: Входное изображение как путь к файлу или numpy массив (BGR/RGB).
        config: Объект конфигурации. Используется по умолчанию, если None.
        confidence_threshold: Переопределить порог уверенности по умолчанию.
        model_path: Путь к весам модели YOLO для инфракрасного обнаружения.
        class_filter: Опциональный список ID классов для фильтрации результатов.

    Returns:
        Список объектов InfraredDetection. Пустой список, если объекты не обнаружены.

    Пример:
        >>> image = cv2.imread('night_scene.png')
        >>> detections = detect_infrared_objects(image)
        >>> for det in detections:
        ...     print(f"Объект: {det.class_name}, ув: {det.confidence:.2f}")
    """
    if config is None:
        config = Config()

    # Разрешить путь к модели
    if model_path is None:
        model_path = config.get_model_path("infrared_detector")
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
        image,
        conf=confidence_threshold or config.infrared_detector.confidence_threshold,
    )
    result = results[0]

    detections = []

    if result.boxes is not None:
        boxes = result.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])

            # Применить фильтр классов, если указан
            if class_filter is not None and class_id not in class_filter:
                continue

            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            class_name = result.names.get(class_id, f"class_{class_id}")

            detections.append(
                InfraredDetection(
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                )
            )

    return detections
