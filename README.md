# Video Analytics Pipeline

Universal, modular video analytics pipeline for maritime navigation (COLREGS compliance).

## Structure

```
colreg/
├── __init__.py           # Package exports
├── config.py             # Centralized configuration
├── boat_detector.py      # Boat detection & cropping (YOLO)
├── binary_classifier.py  # Sailboat classification
├── infrared_detector.py  # Night/infrared detection
├── day_shapes.py         # Day shapes classification
├── lights.py             # Navigation lights classification
├── pipeline.py           # Main orchestrator
└── README.md             # This file
```

## Quick Start

### Full Pipeline

```python
import cv2
from pipeline import VideoAnalyticsPipeline

# Initialize
pipeline = VideoAnalyticsPipeline()

# Load image
image = cv2.imread('frame.jpg')

# Day mode
result = pipeline.process(image, is_night=False)
print(f"Boats: {result.boat_count}, Sailboats: {result.sailboat_count}")

# Night mode
result = pipeline.process(image, is_night=True)
for boat in result.boats:
    if boat.lights_status:
        print(f"Boat status: {boat.lights_status.vessel_type}")
```

### Individual Modules

All modules work independently:

```python
import cv2
from pipeline import (
    detect_and_crop_boats,
    BinaryClassifier,
    classify_day_shapes,
    classify_lights,
    detect_infrared_objects
)

image = cv2.imread('frame.jpg')

# Boat detection
boats = detect_and_crop_boats(image)
for boat in boats:
    print(f"Boat detected: {boat.confidence:.2f}")

# Binary classifier (reuse instance)
classifier = BinaryClassifier()
for boat in boats:
    result = classifier.classify(boat.crop)
    print(f"{result.predicted_class}: {result.confidence:.1f}%")

# Day shapes
statuses = classify_day_shapes(image)
for status in statuses:
    print(f"Vessel type: {status.vessel_type}")

# Lights (night)
lights = classify_lights(image)
for light in lights:
    print(f"Vessel type: {light.vessel_type}")

# Infrared (night)
ir_dets = detect_infrared_objects(image)
for det in ir_dets:
    print(f"IR object: {det.class_name}")
```

## Features

- **Pipeline-agnostic**: All functions work independently
- **Flexible input**: Accept file paths or numpy arrays
- **Configurable**: Override thresholds, paths, parameters
- **Type-safe**: Full type hints and dataclasses
- **No side effects**: Functions don't save files unless explicitly requested

## Configuration

```python
from pipeline import Config

config = Config()

# Override defaults
config.boat_detector.confidence_threshold = 0.7
config.lights.confidence_threshold = 0.6
config.grouping_x_tolerance = 50  # For mast grouping

# Use in pipeline
pipeline = VideoAnalyticsPipeline(config=config)

# Or in individual functions
boats = detect_and_crop_boats(image, config=config)
```

## API Reference

### Boat Detector

```python
detect_and_crop_boats(
    image: Union[str, Path, np.ndarray],
    config: Optional[Config] = None,
    confidence_threshold: Optional[float] = None,
    class_id: Optional[int] = None,
    model_path: Optional[Union[str, Path]] = None
) -> List[BoatDetection]
```

### Binary Classifier

```python
classifier = BinaryClassifier(
    model_path: Optional[Union[str, Path]] = None,
    config: Optional[Config] = None,
    device: Optional[str] = None
)

result = classifier.classify(
    image: Union[str, Path, np.ndarray, Image.Image]
) -> ClassificationResult
```

### Day Shapes

```python
classify_day_shapes(
    image: Union[str, Path, np.ndarray],
    config: Optional[Config] = None,
    confidence_threshold: Optional[float] = None,
    model_path: Optional[Union[str, Path]] = None,
    x_tolerance: Optional[int] = None,
    return_detections: bool = False
) -> List[VesselTypeResult]
```

### Lights

```python
classify_lights(
    image: Union[str, Path, np.ndarray],
    config: Optional[Config] = None,
    confidence_threshold: Optional[float] = None,
    model_path: Optional[Union[str, Path]] = None,
    x_tolerance: Optional[int] = None,
    return_detections: bool = False
) -> List[VesselTypeResult]
```

### Infrared Detector

```python
detect_infrared_objects(
    image: Union[str, Path, np.ndarray],
    config: Optional[Config] = None,
    confidence_threshold: Optional[float] = None,
    model_path: Optional[Union[str, Path]] = None,
    class_filter: Optional[List[int]] = None
) -> List[InfraredDetection]
```

### Full Pipeline

```python
pipeline = VideoAnalyticsPipeline(
    config: Optional[Config] = None,
    classifier_device: Optional[str] = None
)

result = pipeline.process(
    image: Union[str, Path, np.ndarray],
    is_night: bool = False,
    boat_confidence: Optional[float] = None,
    classifier_confidence: Optional[float] = None,
    skip_classification: bool = False
) -> PipelineResult
```

### Night Mode with Separate IR and Visible Images

```python
pipeline = VideoAnalyticsPipeline()

result = pipeline.process_night(
    ir_image: Union[str, Path, np.ndarray],
    visible_image: Union[str, Path, np.ndarray],
    boat_confidence: Optional[float] = None,
    classifier_confidence: Optional[float] = None,
    skip_classification: bool = False,
    bbox_offset: Tuple[int, int] = (0, 0),
    bbox_scale: Tuple[float, float, float, float] = (1.0, 1.5, 1.0, 1.0)
) -> PipelineResult
```

### Visualization

```python
from pipeline import draw_results

pipeline = VideoAnalyticsPipeline()
result = pipeline.process(image, is_night=False)
vis = draw_results(image, result)
cv2.imwrite('output.png', vis)
```

## COLREGS Rules Implemented

### Day Shapes
| Status | Sequence | Description |
|--------|----------|-------------|
| NUC | Ball, Ball | Not Under Command |
| RAM | Ball, Diamond, Ball | Restricted Ability to Maneuver |
| CBD | Cylinder | Constrained by Draft |
| Fishing | Cone down, Cone up | Engaged in Fishing |

### Navigation Lights
| Status | Sequence | Description |
|--------|----------|-------------|
| NUC | Red, Red | Not Under Command |
| RAM | Red, White, Red | Restricted Ability to Maneuver |
| CBD | Red, Red, Red | Constrained by Draft |
| Fishing | Red, White | Engaged in Fishing |
| Trawling | Green, White | Engaged in Trawling |

## Requirements

- Python 3.8+
- ultralytics (YOLO)
- torch, torchvision
- opencv-python
- PIL/Pillow
- numpy

## Installation

```bash
pip install ultralytics torch torchvision opencv-python pillow numpy
```
