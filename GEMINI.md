# GEMINI.md - COLREG Video Analytics Project

## Project Overview
This project is a high-performance video analytics pipeline designed for automated maritime vessel classification according to **COLREG-72** (International Regulations for Preventing Collisions at Sea). It identifies vessels and determines their specific operational status (e.g., Not Under Command, Restricted in Ability to Maneuver) using visual cues like day shapes and navigation lights.

### Key Technologies
- **Python 3.8+**: Primary development language.
- **Ultralytics YOLOv11**: Used for general boat detection, infrared detection, day shapes, and navigation light identification.
- **PyTorch**: Powers the binary classifier (Sailboat vs. Mechanical Vessel).
- **OpenCV (cv2)**: Handles image processing, coordinate transformations, and visualization.
- **NumPy**: Used for efficient array operations and numerical logic.

### Architecture & Priority
The system follows a hierarchical classification logic to determine the `final_vessel_type`:
1.  **NUC** (Not Under Command) - Highest priority.
2.  **RAM** (Restricted in Ability to Maneuver).
3.  **CBD** (Constrained by Draught).
4.  **FISHING**.
5.  **SAIL / MECH** (Determined by binary classification) - Lowest priority.

---

## Directory Structure
- `pipeline.py`: The main orchestrator that coordinates all detection and classification modules.
- `boat_detector.py`: Implements YOLO-based general boat detection (COCO class 8).
- `binary_classifier.py`: Deep learning model for distinguishing between sailing and mechanical vessels.
- `day_shapes.py`: Logic for detecting and grouping day signals (balls, cones, diamonds, cylinders) to infer vessel status.
- `lights.py`: Logic for detecting and grouping navigation lights (red, green, white) for night-time classification.
- `infrared_detector.py`: Specialized YOLO model for object detection in thermal/IR imagery.
- `config.py`: Centralized configuration for model paths, class mappings, and confidence thresholds.
- `models/`: Contains pre-trained weights (`.pt` and `.pth` files).

---

## Building and Running

### Prerequisites
Install dependencies using the following (inferred from imports):
```bash
pip install ultralytics opencv-python torch torchvision pillow numpy
```

### Basic Usage
The primary entry point is the `VideoAnalyticsPipeline` class in `pipeline.py`.

**Day Mode Analysis:**
```python
from pipeline import VideoAnalyticsPipeline
pipeline = VideoAnalyticsPipeline()
result = pipeline.process(image, is_night=False)
```

**Night Mode Analysis (Dual-Sensor):**
```python
# Uses IR for detection and Visible light for navigation light classification
result = pipeline.process_night(ir_image, visible_image)
```

---

## Development Conventions

### Data Models
- Use `@dataclass` for all result containers (e.g., `BoatAnalysisResult`, `VesselTypeResult`).
- Follow the BGR color format consistent with OpenCV.

### Coordinate Systems
- All modules should return bounding boxes in `[x1, y1, x2, y2]` format.
- Grouping logic (in `day_shapes.py` and `lights.py`) uses a vertical mast-based grouping strategy with a configurable `x_tolerance`.

### Model Management
- All model paths must be managed through `Config` and resolved via `config.get_model_path()`.
- Lazy loading is preferred for heavy models (see `VideoAnalyticsPipeline.classifier`).

---

## TODOs & Missing Items
- [ ] **Tests:** No automated test suite found. Implementation of unit tests for `_group_by_mast` and `_classify_group` is recommended.
- [ ] **Dependency File:** `requirements.txt` is mentioned in README but missing from the root.
- [ ] **CLI Wrapper:** A standalone CLI script for processing video files/folders would be beneficial.
