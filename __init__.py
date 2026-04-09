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

from binary_classifier import BinaryClassifier
from boat_detector import detect_and_crop_boats
from config import Config
from day_shapes import VesselType, VesselTypeResult, classify_day_shapes
from infrared_detector import detect_infrared_objects
from lights import VesselTypeResult as LightsVesselTypeResult
from lights import classify_lights
from pipeline import BoatAnalysisResult, PipelineResult, VideoAnalyticsPipeline

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
