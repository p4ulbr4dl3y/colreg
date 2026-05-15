"""Общие типы данных конвейера видеоаналитики: результаты детекции и классификации."""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np


class VesselType:
    """Типы судов в соответствии с МППСС-72.

    Атрибуты:
        - MECHANICAL: судно на ходу, приводимое в движение механической установкой;
        - SAIL: парусное судно;
        - FISHING: судно, занятое ловом рыбы;
        - NUC: судно, не имеющее возможности управляться;
        - RAM: судно, ограниченное в возможности маневрировать;
        - CBD: судно, стесненное своей осадкой.
    """

    MECHANICAL = "MECH"
    SAIL = "SAIL"
    FISHING = "FISH"
    NUC = "NUC"
    RAM = "RAM"
    CBD = "CBD"


@dataclass
class SignalResult:
    """Результат классификации сигналов судна: дневных знаков или огней.

    Атрибуты:
        - vessel_type: тип судна, определенный по сигналам;
        - bbox: координаты ограничивающей рамки группы сигналов [x1, y1, x2, y2];
        - color: цвет BGR для визуализации;
        - signals: список обнаруженных объектов сигналов;
        - sequence: последовательность идентификаторов классов сигналов.
    """

    vessel_type: str
    bbox: List[int]
    color: Tuple[int, int, int]
    signals: List[Any]
    sequence: List[int]

    @property
    def is_known_signal(self) -> bool:
        """Проверяет, является ли сигнал распознанным согласно МППСС-72."""
        return self.vessel_type not in ["Unknown", "Неизвестный сигнал"]

    @property
    def confidence(self) -> float:
        """Вычисляет среднюю уверенность детекции сигналов в группе."""
        if not self.signals:
            return 0.0
        return sum((s.confidence for s in self.signals)) / len(self.signals)


@dataclass
class BoatAnalysisResult:
    """Результат детального анализа отдельного судна.

    Атрибуты:
        - boat_id: уникальный идентификатор судна;
        - crop: фрагмент изображения с обнаруженным судном;
        - bbox: координаты ограничивающей рамки на исходном изображении [x1, y1, x2, y2];
        - detection_confidence: уверенность в обнаружении судна;
        - vessel_type: базовый тип судна, определенный классификатором;
        - vessel_type_confidence: уверенность в определении типа судна в процентах;
        - day_shapes_status: статус судна по дневным знакам;
        - lights_status: статус судна по навигационным огням;
        - infrared_detections: список ИК-детекций, связанных с данным судном.
    """

    boat_id: int
    crop: np.ndarray
    bbox: List[int]
    detection_confidence: float
    vessel_type: str = "MECH"
    vessel_type_confidence: float = 0.0
    day_shapes_status: Optional[SignalResult] = None
    lights_status: Optional[SignalResult] = None
    infrared_detections: List[Any] = field(default_factory=list)

    @property
    def final_vessel_type(self) -> str:
        """Определяет итоговый тип судна с учетом иерархии МППСС-72."""
        if self.day_shapes_status and self.day_shapes_status.is_known_signal:
            return self.day_shapes_status.vessel_type
        if self.lights_status and self.lights_status.is_known_signal:
            return self.lights_status.vessel_type
        return self.vessel_type

    @property
    def final_vessel_type_confidence(self) -> float:
        """Определяет уверенность в итоговом типе судна."""
        if self.day_shapes_status and self.day_shapes_status.is_known_signal:
            return self.day_shapes_status.confidence * 100.0
        if self.lights_status and self.lights_status.is_known_signal:
            return self.lights_status.confidence * 100.0
        return self.vessel_type_confidence


@dataclass
class PipelineResult:
    """Результат работы всего конвейера обработки видео.

    Атрибуты:
        - image: исходное или обработанное изображение;
        - is_night: флаг ночного режима работы;
        - boats: список проанализированных судов;
        - infrared_detections: список всех ИК-детекций на кадре;
        - day_shapes_statuses: результаты классификации дневных знаков;
        - lights_statuses: результаты классификации огней.
    """

    image: np.ndarray
    is_night: bool
    boats: List[BoatAnalysisResult] = field(default_factory=list)
    infrared_detections: List[Any] = field(default_factory=list)
    day_shapes_statuses: List[SignalResult] = field(default_factory=list)
    lights_statuses: List[SignalResult] = field(default_factory=list)

    @property
    def boat_count(self) -> int:
        """Возвращает общее количество обнаруженных судов."""
        return len(self.boats)

    @property
    def sailboat_count(self) -> int:
        """Возвращает количество парусных судов."""
        return sum((1 for b in self.boats if b.vessel_type == VesselType.SAIL))

    @property
    def mechanical_count(self) -> int:
        """Возвращает количество судов с механическим двигателем."""
        return sum(
            (1 for b in self.boats if b.final_vessel_type == VesselType.MECHANICAL)
        )


# Цвета для визуализации классов судов
CLASS_COLORS = {
    VesselType.NUC: (0, 0, 255),
    VesselType.RAM: (255, 0, 255),
    VesselType.CBD: (0, 165, 255),
    VesselType.FISHING: (0, 255, 255),
    VesselType.SAIL: (0, 255, 0),
    VesselType.MECHANICAL: (255, 200, 0),
    "MECH": (255, 200, 0),
    "SAIL": (0, 255, 0),
}
