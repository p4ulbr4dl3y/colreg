"""
Universal Video Analytics Pipeline for Maritime Navigation.

This package provides modular, reusable functions for:
- Boat detection and cropping
- Binary classification (sailboat detection)
- Infrared object detection (night mode)
- Day-shapes classification
- Navigation lights classification
"""

from .binary_classifier import BinaryClassifier
from .boat_detector import detect_and_crop_boats
from .config import Config
from .day_shapes import classify_day_shapes
from .infrared_detector import detect_infrared_objects
from .lights import classify_lights
from .pipeline import VideoAnalyticsPipeline

__all__ = [
    "Config",
    "detect_and_crop_boats",
    "BinaryClassifier",
    "detect_infrared_objects",
    "classify_day_shapes",
    "classify_lights",
    "VideoAnalyticsPipeline",
]
