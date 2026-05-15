"""Классификатор дневных сигнальных фигур для определения статуса судна по МППСС-72."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from colreg_vision.core.config import Config
from colreg_vision.core.types import SignalResult, VesselType


@dataclass
class DayShapeDetection:
    """Данные об обнаруженной дневной фигуре.

    Атрибуты:
        - class_id: идентификатор класса фигуры;
        - class_name: текстовое название фигуры;
        - bbox: координаты ограничивающей рамки;
        - center_x: координата X центра фигуры;
        - center_y: координата Y центра фигуры;
        - confidence: уверенность обнаружения.
    """

    class_id: int
    class_name: str
    bbox: List[int]
    center_x: float
    center_y: float
    confidence: float


DAY_SHAPES_RULES = {
    VesselType.NUC: {
        "sequence": [0, 0],
        "color": (0, 0, 255),
        "description": "Не может управляться - 2 шара",
    },
    VesselType.RAM: {
        "sequence": [0, 3, 0],
        "color": (170, 255, 170),
        "description": "Ограничено в возможности маневрировать - шар-ромб-шар",
    },
    VesselType.CBD: {
        "sequence": [4],
        "color": (0, 165, 255),
        "description": "Стеснено своей осадкой - цилиндр",
    },
    VesselType.FISHING: {
        "sequence": [2, 1],
        "color": (0, 255, 255),
        "description": "Занято ловом рыбы - конусы вершинами вместе",
    },
}


def _group_by_mast(
    detections: List[DayShapeDetection],
    x_tolerance: int = 40,
    max_y_gap_factor: float = 4.0,
    max_area_ratio: float = 4.0,
) -> List[List[DayShapeDetection]]:
    """Группирует обнаруженные фигуры по вертикальным мачтам.

    Аргументы:
        - detections: список обнаруженных фигур;
        - x_tolerance: допуск по горизонтали для отнесения к одной мачте;
        - max_y_gap_factor: максимальный вертикальный разрыв относительно высоты фигур;
        - max_area_ratio: максимальное отношение площадей фигур в одной группе.

    Возвращает:
        список групп фигур, отсортированных по вертикали.
    """
    if not detections:
        return []
    sorted_detections = sorted(detections, key=lambda x: x.center_y)
    groups = []
    current_group = [sorted_detections[0]]

    def get_area(d: DayShapeDetection) -> float:
        return (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])

    def get_height(d: DayShapeDetection) -> float:
        return d.bbox[3] - d.bbox[1]

    for i in range(1, len(sorted_detections)):
        prev = current_group[-1]
        curr = sorted_detections[i]
        x_aligned = abs(curr.center_x - prev.center_x) < x_tolerance
        avg_height = (get_height(prev) + get_height(curr)) / 2
        y_gap = curr.bbox[1] - prev.bbox[3]
        y_close_enough = y_gap < max_y_gap_factor * avg_height
        current_areas = [get_area(d) for d in current_group] + [get_area(curr)]
        area_ratio_ok = max(current_areas) / min(current_areas) <= max_area_ratio
        if x_aligned and y_close_enough and area_ratio_ok:
            current_group.append(curr)
        else:
            groups.append(current_group)
            current_group = [curr]
    groups.append(current_group)
    return groups


def _classify_group(
    group: List[DayShapeDetection], rules: dict = DAY_SHAPES_RULES
) -> SignalResult:
    """Классифицирует состояние судна для отдельной группы фигур.

    Аргументы:
        - group: группа фигур на одной мачте;
        - rules: словарь правил классификации.

    Возвращает:
        результат анализа сигналов.
    """
    sequence = [d.class_id for d in group]
    vessel_type = "Unknown"
    color = (0, 0, 255)
    for vtype, rule in rules.items():
        if sequence == rule["sequence"]:
            vessel_type = vtype
            color = rule["color"]
            break
    x1_min = min((d.bbox[0] for d in group))
    y1_min = min((d.bbox[1] for d in group))
    x2_max = max((d.bbox[2] for d in group))
    y2_max = max((d.bbox[3] for d in group))
    return SignalResult(
        vessel_type=vessel_type,
        bbox=[x1_min, y1_min, x2_max, y2_max],
        color=color,
        signals=group,
        sequence=sequence,
    )


def classify_day_shapes(
    image: Union[str, Path, np.ndarray],
    config: Optional[Config] = None,
    confidence_threshold: Optional[float] = None,
    model_path: Optional[Union[str, Path]] = None,
    x_tolerance: Optional[int] = None,
    return_detections: bool = False,
    model: Optional[YOLO] = None,
) -> Union[List[SignalResult], Tuple[List[SignalResult], List[DayShapeDetection]]]:
    """Классифицирует состояние судна по дневным фигурам.

    Функция обнаруживает фигуры на изображении, группирует их и сопоставляет
    с правилами МППСС-72 для определения статуса судна.

    Аргументы:
        - image: входное изображение;
        - config: объект конфигурации;
        - confidence_threshold: порог уверенности детекции;
        - model_path: путь к весам модели;
        - x_tolerance: допуск группировки по горизонтали;
        - return_detections: флаг возврата списка всех обнаруженных фигур;
        - model: экземпляр модели YOLO.

    Возвращает:
        результаты анализа сигналов или кортеж из результатов и всех детекций.

    Исключения:
        - ValueError: если не удалось загрузить изображение;
        - TypeError: если тип изображения не поддерживается.
    """
    if config is None:
        config = Config()
    if model_path is None:
        model_path = config.get_model_path("day_shapes")
    else:
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = config.base_dir / model_path
    if isinstance(image, (str, Path)):
        image_cv = cv2.imread(str(image))
        if image_cv is None:
            raise ValueError(f"Не удалось загрузить изображение: {image}")
        image = image_cv
    elif not isinstance(image, np.ndarray):
        raise TypeError("Изображение должно быть путём к файлу или numpy массивом")
    if model is None:
        model = YOLO(str(model_path))
    results = model(
        image,
        conf=confidence_threshold or config.day_shapes.confidence_threshold,
        device=config.device,
        verbose=False,
    )
    result = results[0]
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
    groups = _group_by_mast(detections, x_tolerance or config.grouping_x_tolerance)
    statuses = [_classify_group(group) for group in groups]
    if return_detections:
        return (statuses, detections)
    return statuses
