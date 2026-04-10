import os
import re
from pathlib import Path

replacements = [
    (r'from config import', r'from colreg_vision.core.config import'),
    (r'from core_types import', r'from colreg_vision.core.types import'),
    (r'from boat_detector import', r'from colreg_vision.detectors.boat import'),
    (r'from infrared_detector import', r'from colreg_vision.detectors.infrared import'),
    (r'from binary_classifier import', r'from colreg_vision.classifiers.binary import'),
    (r'from day_shapes import', r'from colreg_vision.classifiers.day_shapes import'),
    (r'from lights import', r'from colreg_vision.classifiers.lights import'),
    (r'from pipeline import', r'from colreg_vision.pipeline import'),
]

def update_imports(dir_path):
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                content = file_path.read_text()
                new_content = content
                for old, new in replacements:
                    new_content = re.sub(old, new, new_content)
                if new_content != content:
                    file_path.write_text(new_content)
                    print(f"Updated {file_path}")

update_imports('src')
update_imports('scripts')
update_imports('tests')
