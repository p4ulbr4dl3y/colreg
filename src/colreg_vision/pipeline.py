"""
Главный файл конвейера видеоаналитики для морской навигации.

Координирует все модули для полного анализа судов.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from colreg_vision.classifiers.binary import BinaryClassifier
from colreg_vision.detectors.boat import BoatDetection, detect_and_crop_boats
from colreg_vision.core.config import Config
from colreg_vision.core.types import CLASS_COLORS
from colreg_vision.classifiers.day_shapes import VesselTypeResult, classify_day_shapes
from colreg_vision.detectors.infrared import InfraredDetection, detect_infrared_objects
from colreg_vision.classifiers.lights import VesselTypeResult as LightsVesselTypeResult
from colreg_vision.classifiers.lights import classify_lights


def expand_bbox(
    bbox: List[int],
    image_shape: Tuple[int, ...],
    bbox_scale: Tuple[float, float, float, float],
) -> List[int]:
    """
    Расширить ограничивающий прямоугольник на заданные коэффициенты масштабирования.

    Args:
        bbox: Исходный bbox [x1, y1, x2, y2].
        image_shape: Размеры изображения (height, width, ...).
        bbox_scale: Коэффициенты масштабирования (left, top, right, bottom).

    Returns:
        Расширенный bbox [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
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


@dataclass
class BoatAnalysisResult:
    """Полный результат анализа для одного судна."""

    boat_id: int
    crop: np.ndarray
    bbox: List[int]
    detection_confidence: float

    # Бинарная классификация → тип судна (МППСС-72)
    # 'sailboat' → VesselType.SAIL, 'not_sailboat' → VesselType.MECHANICAL
    vessel_type: str = "MECH"
    vessel_type_confidence: float = 0.0

    # Дневные фигуры (если применимо) - переопределяет тип судна, если обнаружено
    day_shapes_status: Optional[VesselTypeResult] = None

    # Огни (если применимо, ночной режим) - переопределяет тип судна, если обнаружено
    lights_status: Optional[LightsVesselTypeResult] = None

    # Инфракрасные обнаружения (ночной режим)
    infrared_detections: List[InfraredDetection] = field(default_factory=list)

    @property
    def final_vessel_type(self) -> str:
        """
        Получить итоговый тип судна согласно приоритету МППСС-72.

        Приоритет (от высшего к низшему):
        1. NUC (Не может управляться)
        2. RAM (Ограничено в возможности маневрировать)
        3. CBD (Стеснено своей осадкой)
        4. Занято ловом рыбы
        5. Парусное / Механическое (от бинарного классификатора)
        """
        # Дневные фигуры имеют наивысший приоритет
        if self.day_shapes_status and self.day_shapes_status.is_known_signal:
            return self.day_shapes_status.vessel_type

        # Классификация огней для ночного режима
        if self.lights_status and self.lights_status.is_known_signal:
            return self.lights_status.vessel_type

        # По умолчанию от бинарного классификатора
        return self.vessel_type

    @property
    def final_vessel_type_confidence(self) -> float:
        """
        Получить уверенность итоговой классификации.

        Если тип определен по сигналам (фигуры/огни), возвращает среднюю уверенность
        составляющих этот сигнал элементов (в процентах).
        Иначе возвращает уверенность бинарного классификатора.
        """
        if self.day_shapes_status and self.day_shapes_status.is_known_signal:
            return self.day_shapes_status.confidence * 100.0

        if self.lights_status and self.lights_status.is_known_signal:
            return self.lights_status.confidence * 100.0

        return self.vessel_type_confidence


@dataclass
class PipelineResult:
    """Полный результат конвейера для изображения."""

    image: np.ndarray
    is_night: bool

    # Все анализы судов
    boats: List[BoatAnalysisResult] = field(default_factory=list)

    # Инфракрасные обнаружения (ночной режим, полное изображение)
    infrared_detections: List[InfraredDetection] = field(default_factory=list)

    # Дневные фигуры (полное изображение) - типы судов от дневных фигур
    day_shapes_statuses: List[VesselTypeResult] = field(default_factory=list)

    # Огни (полное изображение, ночной режим) - типы судов от огней
    lights_statuses: List[LightsVesselTypeResult] = field(default_factory=list)

    @property
    def boat_count(self) -> int:
        return len(self.boats)

    @property
    def sailboat_count(self) -> int:
        """Подсчитать парусные суда (для обратной совместимости)."""
        return sum(1 for b in self.boats if b.vessel_type == "Парусное судно")

    @property
    def mechanical_count(self) -> int:
        """Подсчитать механические суда (тип по умолчанию)."""
        return sum(
            1
            for b in self.boats
            if b.final_vessel_type == "Судно с механическим двигателем"
        )


class VideoAnalyticsPipeline:
    """
    Полный конвейер видеоаналитики для морской навигации.

    Этот конвейер координирует все модули:
    1. Обнаружение и кадрирование судов (YOLO)
    2. Бинарная классификация (парусное vs непарусное)
    3. Инфракрасное обнаружение (ночной режим)
    4. Классификация дневных фигур (дневной режим)
    5. Классификация навигационных огней (ночной режим)

    Все функции также доступны отдельно для автономного использования.

    Пример:
        >>> pipeline = VideoAnalyticsPipeline()
        >>> image = cv2.imread('frame.jpg')

        >>> # Дневной режим
        >>> result = pipeline.process(image, is_night=False)
        >>> print(f"Суда: {result.boat_count}, Парусные: {result.sailboat_count}")

        >>> # Ночной режим
        >>> result = pipeline.process(image, is_night=True)
        >>> for boat in result.boats:
        ...     if boat.lights_status:
        ...         print(f"Судно {boat.boat_id}: {boat.lights_status.vessel_type}")
    """

    def __init__(
        self, config: Optional[Config] = None, classifier_device: Optional[str] = None
    ):
        """
        Инициализировать конвейер.

        Args:
            config: Объект конфигурации. Используется по умолчанию, если None.
            classifier_device: Устройство для бинарного классификатора ('cuda', 'cpu' или None).
        """
        self.config = config or Config()

        # Инициализировать бинарный классификатор (тяжёлый, загружаем один раз)
        self._classifier = None
        self._classifier_device = classifier_device

    @property
    def classifier(self) -> BinaryClassifier:
        """Ленивая загрузка бинарного классификатора."""
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
        bbox_scale: Tuple[float, float, float, float] = (1.0, 5.0, 1.0, 1.0),
    ) -> PipelineResult:
        """
        Обработать изображение через полный конвейер.

        Args:
            image: Входное изображение как путь к файлу или numpy массив (BGR).
            is_night: Если True, использовать инфракрасное обнаружение и огни.
            boat_confidence: Переопределить порог уверенности обнаружения судов.
            classifier_confidence: Переопределить порог уверенности бинарного классификатора.
            skip_classification: Пропустить шаг бинарной классификации.
            bbox_scale: Коэффициенты масштабирования для расширения bbox (слева, сверху, справа, снизу).
                По умолчанию (1.0, 1.0, 1.0, 1.0) — без масштабирования.

        Returns:
            PipelineResult со всеми результатами анализа.

        Пример:
            >>> image = cv2.imread('frame.jpg')
            >>> result = pipeline.process(image, is_night=False)
            >>> for boat in result.boats:
            ...     print(f"Парусное: {boat.is_sailboat}, ув: {boat.sailboat_confidence:.1f}%")
        """
        # Загрузить изображение, если путь
        if isinstance(image, (str, Path)):
            image_cv = cv2.imread(str(image))
            if image_cv is None:
                raise ValueError(f"Не удалось загрузить изображение: {image}")
            image = image_cv
        elif not isinstance(image, np.ndarray):
            raise TypeError("Изображение должно быть путём к файлу или numpy массивом")

        # Шаг 1: Обнаружить и кадрировать суда
        boat_detections = detect_and_crop_boats(
            image=image, config=self.config, confidence_threshold=boat_confidence
        )

        # Шаг 2: Бинарная классификация для каждого судна → тип судна
        boats = []
        for i, boat_det in enumerate(boat_detections):
            # Рассчитать расширенный bbox
            exp_bbox = expand_bbox(boat_det.bbox, image.shape, bbox_scale)
            exp_x1, exp_y1, exp_x2, exp_y2 = exp_bbox

            # Вырезать с расширенным bbox
            crop = image[exp_y1:exp_y2, exp_x1:exp_x2]

            boat_result = BoatAnalysisResult(
                boat_id=i,
                crop=crop,
                bbox=exp_bbox,
                detection_confidence=boat_det.confidence,
            )

            # Бинарный классификатор: sailboat vs not_sailboat → тип судна
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

        # Инициализировать результат
        result = PipelineResult(image=image, is_night=is_night, boats=boats)

        # Шаг 3: Ночной режим - инфракрасное обнаружение и огни
        if is_night:
            # Инфракрасное обнаружение на полном изображении
            result.infrared_detections = detect_infrared_objects(
                image=image, config=self.config
            )

            # Классификация огней на каждом вырезе судна
            for boat in boats:
                lights_statuses = classify_lights(image=boat.crop, config=self.config)

                # Фильтрация огней по размеру
                valid_statuses = []
                for status in lights_statuses:
                    if not status.is_known_signal:
                        continue

                    signal_w = status.bbox[2] - status.bbox[0]
                    signal_h = status.bbox[3] - status.bbox[1]
                    signal_area = signal_w * signal_h
                    crop_h, crop_w = boat.crop.shape[:2]
                    crop_area = crop_h * crop_w

                    if signal_area < crop_area * 0.3 and signal_w < crop_w * 0.6:
                        valid_statuses.append(status)

                if valid_statuses:
                    boat.lights_status = valid_statuses[0]

        # Шаг 4: Дневной режим - дневные фигуры на каждом вырезе судна
        else:
            for boat in boats:
                day_statuses = classify_day_shapes(image=boat.crop, config=self.config)

                # Фильтрация фигур по размеру
                valid_statuses = []
                for status in day_statuses:
                    # Если статус неизвестен, пропускаем проверку размера
                    if not status.is_known_signal:
                        continue

                    # Фигуры не должны занимать слишком большую часть судна
                    signal_w = status.bbox[2] - status.bbox[0]
                    signal_h = status.bbox[3] - status.bbox[1]
                    signal_area = signal_w * signal_h

                    crop_h, crop_w = boat.crop.shape[:2]
                    crop_area = crop_h * crop_w

                    # 1. Сигнал не должен занимать слишком большую часть площади
                    # 2. Сигнал не должен быть шире 60% от ширины судна
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
        """
        Обработать ночной режим с раздельными ИК и видимыми изображениями.

        Использует ИК-изображение для обнаружения судов (модель infrared.pt), затем классифицирует
        навигационные огни на вырезах видимого изображения по тем же координатам.
        Результаты визуализируются на ИК-изображении.

        Args:
            ir_image: Инфракрасное/тепловое изображение для обнаружения судов.
            visible_image: Изображение в видимом свете для классификации огней.
            boat_confidence: Переопределить порог уверенности обнаружения судов.
            classifier_confidence: Переопределить порог уверенности бинарного классификатора.
            skip_classification: Пропустить шаг бинарной классификации.
            bbox_offset: Опциональное смещение (dx, dy) для коррекции выравнивания между
                ИК и видимым датчиками. По умолчанию (0, 0).
            bbox_scale: Коэффициенты масштабирования для расширения bbox (слева, сверху, справа, снизу).
                По умолчанию (1.0, 1.5, 1.0, 1.0) - расширяет верх на 50% для захвата мачты/огней.

        Returns:
            PipelineResult с результатами анализа.

        Пример:
            >>> ir = cv2.imread('thermal_frame.png')
            >>> vis = cv2.imread('visible_frame.png')
            >>> result = pipeline.process_night(ir, vis, bbox_scale=(1.0, 2.0, 1.0, 1.0))
            >>> for boat in result.boats:
            ...     if boat.lights_status:
            ...         print(f"Судно {boat.boat_id}: {boat.lights_status.vessel_type}")
        """
        # Загрузить изображения, если пути
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

        # Шаг 1: Обнаружить суда на ИК-изображении с помощью модели infrared.pt

        ir_detections_raw = detect_infrared_objects(
            image=ir_image,
            config=self.config,
            confidence_threshold=boat_confidence,
        )

        # Преобразовать InfraredDetection в BoatDetection для совместимости
        boat_detections = []
        for i, ir_det in enumerate(ir_detections_raw):
            x1, y1, x2, y2 = ir_det.bbox
            crop = ir_image[y1:y2, x1:x2].copy()
            boat_detections.append(
                BoatDetection(
                    crop=crop,
                    bbox=ir_det.bbox,
                    confidence=ir_det.confidence,
                    crop_id=i,
                )
            )

        # Шаг 2: Бинарная классификация на ИК-вырезах, огни на видимых вырезах
        boats = []
        for i, boat_det in enumerate(boat_detections):
            # Рассчитать расширенный bbox (расширяем вверх и немного наружу)
            exp_bbox = expand_bbox(boat_det.bbox, ir_image.shape, bbox_scale)
            exp_x1, exp_y1, exp_x2, exp_y2 = exp_bbox

            # Скорректировать для выравнивания видимого изображения
            adj_bbox = [
                max(0, exp_x1 + bbox_offset[0]),
                max(0, exp_y1 + bbox_offset[1]),
                max(0, exp_x2 + bbox_offset[0]),
                max(0, exp_y2 + bbox_offset[1]),
            ]

            # Вырезать из ИК-изображения для бинарной классификации (используем расширенный bbox)
            ir_crop = ir_image[exp_y1:exp_y2, exp_x1:exp_x2]

            # Вырезать из видимого изображения со скорректированным bbox для классификации огней
            adj_crop = visible_image[
                adj_bbox[1] : adj_bbox[3], adj_bbox[0] : adj_bbox[2]
            ]

            boat_result = BoatAnalysisResult(
                boat_id=i,
                crop=adj_crop,  # видимый вырез для огней
                bbox=exp_bbox,  # расширенные ИК-координаты для рисования на ИК-изображении
                detection_confidence=boat_det.confidence,
            )

            # Бинарный классификатор на ИК-вырезе
            if not skip_classification and boat_det.crop.size > 0:
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

        # Инициализировать результат - использовать ИК-изображение для визуализации
        result = PipelineResult(image=ir_image, is_night=True, boats=boats)

        # Сохранить ИК-обнаружения (уже вычислены в Шаге 1)
        result.infrared_detections = ir_detections_raw

        # Шаг 4: Классификация огней на видимых вырезах
        for boat in boats:
            if boat.crop.size > 0:
                lights_statuses = classify_lights(image=boat.crop, config=self.config)

                valid_statuses = []
                for status in lights_statuses:
                    if not status.is_known_signal:
                        continue

                    signal_w = status.bbox[2] - status.bbox[0]
                    signal_h = status.bbox[3] - status.bbox[1]
                    signal_area = signal_w * signal_h
                    crop_h, crop_w = boat.crop.shape[:2]
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
    """
    Нарисовать визуализацию результатов конвейера на исходном изображении.

    Рисует ограничивающие прямоугольники с типами судов для каждого обнаруженного судна.
    Метки отображаются внутри рамки сбоку для лучшей видимости при расширенных bbox.

    Args:
        image: Исходное изображение (BGR).
        result: PipelineResult с результатами анализа.
        thickness: Толщина линии прямоугольника.
        font_scale: Масштаб шрифта для меток.

    Returns:
        Изображение с нарисованными результатами.

    Пример:
        >>> pipeline = VideoAnalyticsPipeline()
        >>> result = pipeline.process(image, is_night=False)
        >>> vis = draw_results(image, result)
        >>> cv2.imwrite('output.png', vis)
    """
    output = image.copy()

    for boat in result.boats:
        x1, y1, x2, y2 = boat.bbox

        # Получить тип судна (с приоритетом от day_shapes/lights)
        label = boat.final_vessel_type

        # Определить цвет на основе типа
        color = CLASS_COLORS.get(label, (255, 255, 255))  # Белый по умолчанию

        # Переопределить цвет, если есть специфический цвет от сигнала
        if boat.day_shapes_status and boat.day_shapes_status.is_known_signal:
            color = boat.day_shapes_status.color
        elif boat.lights_status and boat.lights_status.is_known_signal:
            color = boat.lights_status.color

        # Нарисовать ограничивающий прямоугольник
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        # Подготовить текст метки
        confidence_str = f"{boat.final_vessel_type_confidence:.0f}%"
        full_label = f"{label} #{boat.boat_id} ({confidence_str})"

        # Получить размер текста
        (label_w, label_h), baseline = cv2.getTextSize(
            full_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Рассчитать позицию метки (внутри рамки, сверху слева с небольшим отступом)
        text_x = x1 + thickness + 2
        text_y = y1 + label_h + thickness + 2

        # Проверить, не выходит ли метка за нижнюю границу рамки
        if text_y > y2:
            text_y = y2 - baseline

        # Нарисовать фон метки
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

        # Нарисовать текст метки
        cv2.putText(
            output,
            full_label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255) if sum(color) < 400 else (0, 0, 0),  # Контрастный текст
            thickness,
        )

    return output
