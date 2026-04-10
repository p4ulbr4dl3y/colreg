# AGENTS.md

## Running Commands

**After installing (`pip install -e .`), run without PYTHONPATH:**
```bash
python scripts/mqtt_node.py    # MQTT production node
pytest tests/test_pipeline.py -v  # Run tests
```

## Package Structure

- Source: `src/colreg_vision/`
- Main class: `colreg_vision.pipeline.VideoAnalyticsPipeline`
- Entry point: `pipeline.process(image, is_night=False)` or `pipeline.process_night(ir_image, visible_image)`

## Development

- **Install:** `pip install -e .` (editable install from pyproject.toml)
- **Linting:** `ruff check src/`
- **Typecheck:** `PYTHONPATH=src mypy src/` (expects missing stubs warnings for torch/ultralytics - this is normal)
- **Dependencies:** `pip install -r requirements.txt`
- **Python:** 3.8+

## Testing

Test images live in:
- `test_images/day/*.png|jpg|webp`
- `test_images/night/{category}/{ir.png,normal.png}`

Run single test:
```bash
pytest tests/test_pipeline.py::test_day_mode_classifications -v -k "cbd"
```

## Production

MQTT topics:
- Subscribe: `colreg/vision/command` (JSON with `request_id`, `action`, `source`, `is_night`)
- Publish: `colreg/vision/result`

Run simulator for local testing (requires broker):
```bash
# Terminal 1: Start amqtt broker
amqtt -c amqtt.yaml &

# Terminal 2: Run simulator
python scripts/mqtt_simulate.py

# Or run node
python scripts/mqtt_node.py
```