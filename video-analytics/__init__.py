"""
Universal Video Analytics Pipeline for Maritime Navigation.

This package provides modular, reusable functions for:
- Boat detection and cropping
- Binary classification (sailboat vs mechanical)
- Infrared object detection (night mode)
- Day-shapes classification (COLREGS 72 vessel types)
- Navigation lights classification (COLREGS 72 vessel types)

COLREGS 72 Vessel Types (МППСС-72):
- Судно с механическим двигателем (Mechanical vessel)
- Парусное судно (Sail vessel)
- Судно, занятое ловом рыбы (Fishing vessel)
- Судно, лишённое возможности управляться (NUC)
- Судно, ограниченное в возможности маневрировать (RAM)
- Судно, стеснённое своей осадкой (CBD)
- Судно, занимающееся тралением (Trawling)
"""

from .binary_classifier import BinaryClassifier
from .boat_detector import detect_and_crop_boats
from .config import Config
from .day_shapes import classify_day_shapes, VesselType, VesselTypeResult
from .infrared_detector import detect_infrared_objects
from .lights import classify_lights, VesselType as LightsVesselType, VesselTypeResult as LightsVesselTypeResult
from .pipeline import VideoAnalyticsPipeline, BoatAnalysisResult, PipelineResult

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
    "BoatAnalysisResult",
    "PipelineResult",
]
