# COLREG Video Analytics Pipeline

High-performance video analytics pipeline designed for automated maritime vessel classification according to **COLREG-72** (International Regulations for Preventing Collisions at Sea).

## Overview

This system processes visual data (visible light and infrared) to identify vessels and determine their operational status using hierarchical rule-based logic and deep learning models. It acts as the core Vision Node for marine autonomous systems or driver-assistance setups.

### Key Features
- **General Vessel Detection**: Primary object detection via YOLO.
- **Binary Classification**: Differentiates between sailing (`SAIL`) and mechanically propelled (`MECH`) vessels.
- **Day Shapes Classification**: Identifies COLREG day signals (balls, cones, diamonds, cylinders).
- **Navigation Lights Classification**: Identifies COLREG night signals (red, green, white).
- **Night Mode (Dual-Sensor Fusion)**: Leverages IR/Thermal imagery for vessel detection and visible light for navigation lights.
- **Production Integration**: Ready-to-deploy MQTT interface (`mqtt_node.py`) for Radar-Slaved (Slew-to-Cue) architectures.
- **Automated Validation**: Comprehensive `pytest` test suite verifying both day and night operational constraints.

---

## Supported Vessel Types (COLREG-72)

Classification follows a strict priority hierarchy (1 = Highest Priority):

| Priority | Type | Description | Day Signal | Night Signal |
|:---:|:---|:---|:---|:---|
| 1 | **NUC** | Not Under Command | 2 balls | 2 red lights |
| 2 | **RAM** | Restricted in Ability to Maneuver | ball-diamond-ball | red-white-red |
| 3 | **CBD** | Constrained by Draught | cylinder | 3 red lights |
| 4 | **FISHING** | Engaged in fishing | 2 cones (apex together)| red-white |
| 5 | **SAIL** | Sailing vessel | — | — |
| 6 | **MECH** | Power-driven vessel | — | — |

*Note: Types 5 and 6 are determined via the binary classifier if no higher-priority COLREG signals are detected.*

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch (CUDA recommended)

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Production Integration (MQTT / Slew-to-Cue)

For integration into actual vessel hardware (e.g., radar-slaved PTZ cameras), use the provided MQTT adapter.

### Start the Vision Node
The node initializes models and connects to the specified MQTT broker:
```bash
python mqtt_node.py
```

### Communication Protocol
The node subscribes to commands and publishes results via JSON payloads.

**Input Command Topic:** `colreg/vision/command`
```json
{
   "request_id": "req-8842",
   "action": "analyze",
   "source": "path/to/frame.jpg",
   "is_night": false
}
```

**Output Result Topic:** `colreg/vision/result`
```json
{
  "request_id": "req-8842",
  "status": "success",
  "is_night": false,
  "boat_count": 1,
  "boats": [
    {
      "boat_id": 0,
      "bbox": [96, 0, 197, 174],
      "vessel_type": "SAIL",
      "confidence": 99.21
    }
  ],
  "processing_time_ms": 55.4
}
```

*To test the integration locally, run `python mqtt_simulate.py` while the node is active.*

---

## Core Pipeline API (Python)

### Day Mode Analysis
```python
import cv2
from pipeline import VideoAnalyticsPipeline

pipeline = VideoAnalyticsPipeline()
image = cv2.imread('frame.jpg')

result = pipeline.process(image, is_night=False)

for boat in result.boats:
    print(f"[{boat.boat_id}] Type: {boat.final_vessel_type} (Conf: {boat.final_vessel_type_confidence:.1f}%)")
```

### Night Mode Analysis (Dual-Sensor)
Night mode uses the thermal camera for robust detection and the visible camera for light classification.
```python
ir_image = cv2.imread('thermal_frame.png')
visible_image = cv2.imread('visible_frame.png')

result = pipeline.process_night(ir_image, visible_image)
```

---

## Project Structure & Architecture

```text
.
├── config.py                # Centralized configuration (model paths, thresholds)
├── core_types.py            # Shared data structures and constants (VesselType)
├── pipeline.py              # Core orchestrator and bounding box scaling logic
├── boat_detector.py         # General YOLO vessel detection
├── binary_classifier.py     # EfficientNet Sail vs. Mech classification
├── day_shapes.py            # COLREG day shapes logic
├── lights.py                # COLREG navigation lights logic
├── mqtt_node.py             # Production MQTT interface
├── mqtt_simulate.py         # Testing script for MQTT integration
├── tests/                   # Pytest automation suite
└── models/                  # Pre-trained weights (.pt / .pth)
    └── unused/              # Deprecated models
```

---

## Testing

The project includes an automated test suite verifying classification logic against standard day and night scenarios.

```bash
# Run all tests
pytest tests/test_pipeline.py -v
```
