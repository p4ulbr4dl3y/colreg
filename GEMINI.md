# Project Context: COLREG Video Analytics

High-performance video analytics pipeline for automated maritime vessel classification according to **COLREG-72** (International Regulations for Preventing Collisions at Sea).

## System Overview
The system processes visual data (visible and infrared) to detect vessels and determine their operational status using rule-based hierarchical logic and deep learning models. It acts as a Vision Node for maritime autonomous systems or navigator assistance systems.

### Key Features
- **Vessel Detection**: Primary object detection via YOLO (`models/boat_detector.pt`).
- **Binary Classification**: Separation into `SAIL` and `MECH` (Mechanical) vessels.
- **Day Shapes Classification**: Recognition of COLREG day signals (balls, cones, diamonds, cylinders).
- **Navigation Lights Classification**: Recognition of night signals (red, green, white lights).
- **Night Mode (Sensor Fusion)**: IR/Thermal images for reliable vessel detection and visible spectrum for light classification.
- **Production Integration**: MQTT-based Vision Node (`scripts/mqtt_node.py`) for Slew-to-Cue integration.

---

## Architectural Context

### Core Pipeline (`src/colreg_vision/pipeline.py`)
The `VideoAnalyticsPipeline` is the central orchestrator. It implements the COLREG priority hierarchy:
1. **NUC** (Not Under Command) - Highest Priority
2. **RAM** (Restricted in Ability to Maneuver)
3. **CBD** (Constrained By Draught)
4. **FISHING** (Engaged in Fishing)
5. **SAIL / MECH** (Determined by binary classifier if no signals detected)

### Modular Design
- **Detectors** (`src/colreg_vision/detectors/`): YOLO-based models for finding boats in visible or IR frames.
- **Classifiers** (`src/colreg_vision/classifiers/`): Specialized modules for state identification (binary, day shapes, lights).
- **Core** (`src/colreg_vision/core/`): Centralized types (`types.py`) and configuration (`config.py`).

### Production Workflow (MQTT)
The system operates as an asynchronous service.
- **Subscribes**: `colreg/vision/command` (JSON with image path/URL and mode).
- **Publishes**: `colreg/vision/result` (JSON with detected boats, types, and confidence).

---

## Operational Mandates & Conventions

### Development Standards
- **Type Safety**: Use `src/colreg_vision/core/types.py` for all pipeline data structures (`BoatAnalysisResult`, `PipelineResult`).
- **Configuration**: Never hardcode paths or thresholds. Use `src/colreg_vision/core/config.py`.
- **Error Handling**: The pipeline must be resilient to corrupt frames or missing models.

### Model & Data Management
- **Models**: Production models are stored in `models/`. Legacy or experimental models belong in `models/unused/`.
- **Test Images**: Use `test_images/day/` and `test_images/night/` for manual and automated verification.
- **Datasets**: `data/` directory contains raw and validated datasets. Do not commit large binary data files to the repository.

### Testing Protocol
- **Empirical Validation**: Any change to classification logic MUST be verified against the existing test suite (`pytest tests/test_pipeline.py`).
- **Mode Coverage**: Ensure changes are tested for both `is_night=True` and `is_night=False` scenarios.

### MQTT Integration
- When modifying `scripts/mqtt_node.py`, verify compatibility with the `amqtt` broker and ensure the JSON schema matches the protocol defined in `README.md`.
- Use `scripts/mqtt_simulate.py` for integration testing.

---

## Key Symbols
- `VideoAnalyticsPipeline`: Main entry point for processing.
- `BoatAnalysisResult`: Data structure containing per-vessel classification results.
- `VesselType`: Enum defining all supported COLREG statuses.
