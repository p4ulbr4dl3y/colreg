"""
Модуль конфигурации для конвейера видеоаналитики.

Централизованная конфигурация для путей к моделям, маппингов классов и порогов.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class ModelConfig:
    """Конфигурация модели с путями и параметрами."""

    path: str
    confidence_threshold: float = 0.5


@dataclass
class Config:
    """
    Централизованная конфигурация для конвейера видеоаналитики.

    Все пути относительны директории video-analytics.
    """

    # Базовая директория (video-analytics)
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent
    )

    # ==================== ПУТИ К МОДЕЛЯМ ====================

    # Обнаружение судов (YOLO)
    boat_detector: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            path="models/boat_detector.pt", confidence_threshold=0.5
        )
    )

    # Бинарный классификатор (парусное vs непарусное)
    binary_classifier: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            path="models/binary_classifier.pth", confidence_threshold=0.5
        )
    )

    # Инфракрасное обнаружение (ночной режим)
    infrared_detector: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            path="models/infrared_detector.pt", confidence_threshold=0.25
        )
    )

    # Классификация дневных фигур
    day_shapes: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            path="models/day_shapes.pt", confidence_threshold=0.5
        )
    )

    # Классификация огней
    lights: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            path="models/lights.pt", confidence_threshold=0.5
        )
    )

    # ==================== МАППИНГИ КЛАССОВ ====================

    # Маппинг классов дневных фигур
    day_shapes_classes: Dict[int, str] = field(
        default_factory=lambda: {
            0: "ball",
            1: "cone_up",
            2: "cone_down",
            3: "diamond",
            4: "cylinder",
        }
    )

    # Маппинг классов огней
    lights_classes: Dict[int, str] = field(
        default_factory=lambda: {0: "white", 1: "red", 2: "green"}
    )

    # Имена классов бинарного классификатора (загружаются из checkpoint)
    binary_classifier_classes: List[str] = field(
        default_factory=lambda: ["not_sailboat", "sailboat"]
    )

    # ==================== ПАРАМЕТРЫ ОБНАРУЖЕНИЯ ====================

    # ID класса судна в наборе данных COCO
    boat_class_id: int = 8

    # Толерантность группировки для дневных фигур и огней (пиксели)
    grouping_x_tolerance: int = 40

    # Размер изображения для бинарного классификатора
    classifier_image_size: int = 224

    # Нормализация для бинарного классификатора
    classifier_normalize_mean: List[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406]
    )
    classifier_normalize_std: List[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225]
    )

    def get_model_path(self, model_name: str) -> Path:
        """Получить абсолютный путь для модели."""
        model_config = getattr(self, model_name)
        if isinstance(model_config, ModelConfig):
            return self.base_dir / model_config.path
        raise ValueError(f"Unknown model: {model_name}")
