"""
Универсальный конвейер видеоаналитики для морской навигации.

Этот пакет предоставляет модульные, переиспользуемые функции для:
- Обнаружения и кадрирования судов
- Бинарной классификации (парусное vs механическое)
- Инфракрасного обнаружения объектов (ночной режим)
- Классификации дневных фигур (типы судов МППСС-72)
- Классификации навигационных огней (типы судов МППСС-72)

Типы судов МППСС-72:
- Судно с механическим двигателем (Mechanical vessel)
- Парусное судно (Sail vessel)
- Судно, занятое ловом рыбы (Fishing vessel)
- Судно, лишённое возможности управляться (NUC)
- Судно, ограниченное в возможности маневрировать (RAM)
- Судно, стеснённое своей осадкой (CBD)
"""

from colreg_vision.classifiers.binary import BinaryClassifier
from colreg_vision.detectors.boat import detect_and_crop_boats
from colreg_vision.core.config import Config
from colreg_vision.classifiers.day_shapes import (
    VesselType,
    VesselTypeResult,
    classify_day_shapes,
)
from colreg_vision.detectors.infrared import detect_infrared_objects
from colreg_vision.classifiers.lights import VesselTypeResult as LightsVesselTypeResult
from colreg_vision.classifiers.lights import classify_lights
from colreg_vision.pipeline import (
    BoatAnalysisResult,
    PipelineResult,
    VideoAnalyticsPipeline,
)

__all__ = [
    "Config",
    "detect_and_crop_boats",
    "BinaryClassifier",
    "detect_infrared_objects",
    "classify_day_shapes",
    "classify_lights",
    "VideoAnalyticsPipeline",
    "VesselType",
    "VesselTypeResult",
    "LightsVesselTypeResult",
    "BoatAnalysisResult",
    "PipelineResult",
]
