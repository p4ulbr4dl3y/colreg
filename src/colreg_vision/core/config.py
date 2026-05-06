from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Конфигурация отдельной модели нейронной сети.

    Атрибуты:
        path: путь к файлу модели относительно базовой директории;
        confidence_threshold: порог уверенности для детекции или классификации.
    """
    path: str
    confidence_threshold: float = 0.5


@dataclass
class Config:
    """Глобальная конфигурация системы видеоаналитики.

    Содержит пути к моделям, настройки классификаторов и параметры обработки изображений.

    Атрибуты:
        base_dir: корневая директория проекта;
        boat_detector: параметры модели детекции судов;
        binary_classifier: параметры модели классификации типа судна;
        infrared_detector: параметры модели детекции в инфракрасном диапазоне;
        day_shapes: параметры модели распознавания дневных фигур;
        lights: параметры модели распознавания навигационных огней;
        day_shapes_classes: словарь соответствия идентификаторов и названий дневных фигур;
        lights_classes: словарь соответствия идентификаторов и цветов огней;
        binary_classifier_classes: список классов бинарного классификатора;
        boat_class_id: идентификатор класса судна в основной модели детекции;
        grouping_x_tolerance: допуск по оси X для группировки объектов;
        classifier_image_size: размер изображения для подачи в классификатор;
        classifier_normalize_mean: средние значения для нормализации изображения;
        classifier_normalize_std: стандартные отклонения для нормализации изображения;
        device: вычислительное устройство;
        use_tracker: флаг использования трекера объектов;
        tracker_type: тип используемого трекера.
    """
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent
    )
    boat_detector: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            path="models/boat_detector.pt", confidence_threshold=0.5
        )
    )
    binary_classifier: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            path="models/binary_classifier.pth", confidence_threshold=0.6
        )
    )
    infrared_detector: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            path="models/infrared_detector.pt", confidence_threshold=0.25
        )
    )
    day_shapes: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            path="models/day_shapes.pt", confidence_threshold=0.88
        )
    )
    lights: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            path="models/lights.pt", confidence_threshold=0.6
        )
    )
    day_shapes_classes: Dict[int, str] = field(
        default_factory=lambda: {
            0: "ball",
            1: "cone_up",
            2: "cone_down",
            3: "diamond",
            4: "cylinder",
        }
    )
    lights_classes: Dict[int, str] = field(
        default_factory=lambda: {0: "white", 1: "red", 2: "green"}
    )
    binary_classifier_classes: List[str] = field(
        default_factory=lambda: ["not_sailboat", "sailboat"]
    )
    boat_class_id: int = 8
    grouping_x_tolerance: int = 40
    classifier_image_size: int = 224
    classifier_normalize_mean: List[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406]
    )
    classifier_normalize_std: List[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225]
    )
    device: Optional[str] = None
    use_tracker: bool = True
    tracker_type: str = "botsort.yaml"

    def get_model_path(self, model_name: str) -> Path:
        """Возвращает абсолютный путь к файлу модели.

        Аргументы:
            model_name: название атрибута конфигурации модели.

        Возвращает:
            Абсолютный путь к файлу модели.

        Исключения:
            ValueError: если указано неизвестное название модели.
        """
        model_config = getattr(self, model_name)
        if isinstance(model_config, ModelConfig):
            return self.base_dir / model_config.path
        raise ValueError(f"Unknown model: {model_name}")
