"""
Модуль классификации дневных фигур.

Классифицирует тип судна по дневным фигурам (шары, конусы, ромбы, цилиндры).
Реализует правила МППСС для дневных сигналов.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from config import Config
from core_types import VesselType


@dataclass
class DayShapeDetection:
    """Представляет обнаруженную дневную фигуру."""

    class_id: int
    class_name: str
    bbox: List[int]  # [x1, y1, x2, y2]
    center_x: float
    center_y: float
    confidence: float


@dataclass
class VesselTypeResult:
    """Классифицированный тип судна по дневным фигурам."""

    vessel_type: str  # Тип судна согласно МППСС-72
    bbox: List[int]  # Объединённый ограничивающий прямоугольник для группы
    color: Tuple[int, int, int]  # Цвет BGR для визуализации
    shapes: List[DayShapeDetection]  # Составляющие фигуры
    sequence: List[int]  # Последовательность классов сверху вниз

    @property
    def is_known_signal(self) -> bool:
        return self.vessel_type not in ["Unknown", "Неизвестный сигнал"]


# Правила дневных фигур МППСС → типы судов
# Цвета в формате BGR, выбраны для видимости на дневных и ночных изображениях
DAY_SHAPES_RULES = {
    # NUC: Не может управляться - Шар, Шар
    VesselType.NUC: {
        "sequence": [0, 0],
        "color": (0, 0, 255),  # Красный
        "description": "Не может управляться - 2 шара",
    },
    # RAM: Ограничено в возможности маневрировать - Шар, Ромб, Шар
    VesselType.RAM: {
        "sequence": [0, 3, 0],
        "color": (170, 255, 170),  # Маджента/Фиолетовый
        "description": "Ограничено в возможности маневрировать - шар-ромб-шар",
    },
    # CBD: Стеснено своей осадкой - Цилиндр
    VesselType.CBD: {
        "sequence": [4],
        "color": (0, 165, 255),  # Оранжевый
        "description": "Стеснено своей осадкой - цилиндр",
    },
    # Занято ловом рыбы - Конус вниз, Конус вверх (вершинами вместе)
    VesselType.FISHING: {
        "sequence": [2, 1],  # cone_down, cone_up
        "color": (0, 255, 255),  # Циан
        "description": "Занято ловом рыбы - конусы вершинами вместе",
    },
}


def _group_by_mast(
    detections: List[DayShapeDetection], x_tolerance: int = 40
) -> List[List[DayShapeDetection]]:
    """
    Сгруппировать обнаружения по мачте (вертикальное выравнивание).

    Args:
        detections: Список обнаруженных фигур.
        x_tolerance: Максимальное горизонтальное расстояние для считания одной мачтой.

    Returns:
        Список групп, каждая группа содержит фигуры на одной мачте.
    """
    if not detections:
        return []

    # Сортировать по вертикальной позиции (сверху вниз)
    sorted_detections = sorted(detections, key=lambda x: x.center_y)

    groups = []
    current_group = [sorted_detections[0]]

    for i in range(1, len(sorted_detections)):
        prev = current_group[-1]
        curr = sorted_detections[i]

        # Проверить, на одной ли мачте (похожая позиция X)
        if abs(curr.center_x - prev.center_x) < x_tolerance:
            current_group.append(curr)
        else:
            # Новая мачта
            groups.append(current_group)
            current_group = [curr]

    groups.append(current_group)
    return groups


def _classify_group(
    group: List[DayShapeDetection], rules: dict = DAY_SHAPES_RULES
) -> VesselTypeResult:
    """
    Классифицировать тип судна по последовательности фигур.

    Args:
        group: Список фигур на одной мачте.
        rules: Словарь правил МППСС.

    Returns:
        VesselTypeResult с результатом классификации.
    """
    # Получить последовательность (сверху вниз)
    sequence = [d.class_id for d in group]

    # Сопоставить с правилами
    vessel_type = "Unknown"
    color = (0, 0, 255)  # Красный (неизвестный)

    for vtype, rule in rules.items():
        if sequence == rule["sequence"]:
            vessel_type = vtype
            color = rule["color"]
            break

    # Рассчитать объединённый ограничивающий прямоугольник
    x1_min = min(d.bbox[0] for d in group)
    y1_min = min(d.bbox[1] for d in group)
    x2_max = max(d.bbox[2] for d in group)
    y2_max = max(d.bbox[3] for d in group)

    return VesselTypeResult(
        vessel_type=vessel_type,
        bbox=[x1_min, y1_min, x2_max, y2_max],
        color=color,
        shapes=group,
        sequence=sequence,
    )


def classify_day_shapes(
    image: Union[str, Path, np.ndarray],
    config: Optional[Config] = None,
    confidence_threshold: Optional[float] = None,
    model_path: Optional[Union[str, Path]] = None,
    x_tolerance: Optional[int] = None,
    return_detections: bool = False,
) -> Union[
    List[VesselTypeResult], Tuple[List[VesselTypeResult], List[DayShapeDetection]]
]:
    """
    Классифицировать тип судна по дневным фигурам.

    Эта функция не зависит от конвейера — принимает любое изображение и возвращает
    классифицированные типы судов согласно дневным сигналам МППСС.

    Args:
        image: Входное изображение как путь к файлу или numpy массив (BGR).
        config: Объект конфигурации. Используется по умолчанию, если None.
        confidence_threshold: Переопределить порог уверенности по умолчанию.
        model_path: Путь к весам модели YOLO.
        x_tolerance: Горизонтальная толерантность для группировки по мачте (пиксели).
        return_detections: Если True, также вернуть сырые обнаружения.

    Returns:
        Список объектов VesselTypeResult. Если return_detections=True,
        также возвращает список сырых объектов DayShapeDetection.

    Пример:
        >>> image = cv2.imread('vessel.png')
        >>> types = classify_day_shapes(image)
        >>> for vtype in types:
        ...     print(f"Тип судна: {vtype.vessel_type}")
    """
    if config is None:
        config = Config()

    # Разрешить путь к модели
    if model_path is None:
        model_path = config.get_model_path("day_shapes")
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
        image, conf=confidence_threshold or config.day_shapes.confidence_threshold
    )
    result = results[0]

    # Извлечь обнаружения
    detections = []
    if result.boxes is not None:
        boxes = result.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])

            detections.append(
                DayShapeDetection(
                    class_id=class_id,
                    class_name=config.day_shapes_classes.get(
                        class_id, f"class_{class_id}"
                    ),
                    bbox=[x1, y1, x2, y2],
                    center_x=(x1 + x2) / 2,
                    center_y=(y1 + y2) / 2,
                    confidence=conf,
                )
            )

    # Сгруппировать по мачте и классифицировать
    groups = _group_by_mast(detections, x_tolerance or config.grouping_x_tolerance)
    statuses = [_classify_group(group) for group in groups]

    if return_detections:
        return statuses, detections
    return statuses
