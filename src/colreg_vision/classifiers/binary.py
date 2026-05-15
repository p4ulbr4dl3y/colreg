"""Бинарный классификатор судов: разделение на парусные и моторные по изображению."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from colreg_vision.core.config import Config


@dataclass
class ClassificationResult:
    """Результат бинарной классификации судна.

    Атрибуты:
        - predicted_class: название предсказанного класса;
        - confidence: уверенность модели в процентах;
        - sailboat_probability: вероятность того, что судно парусное;
        - not_sailboat_probability: вероятность того, что судно не является парусным.
    """

    predicted_class: str
    confidence: float
    sailboat_probability: float
    not_sailboat_probability: float

    @property
    def is_sailboat(self) -> bool:
        """Проверяет, является ли судно парусным.

        Возвращает:
            истина, если судно классифицировано как парусное.
        """
        return self.predicted_class == "sailboat"


class BinaryClassifier:
    """Классификатор для разделения судов на парусные и моторные.

    Использует модель EfficientNet для определения типа судна по его изображению.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[Config] = None,
        device: Optional[str] = None,
    ):
        """Инициализирует классификатор.

        Аргументы:
            - model_path: путь к файлу модели;
            - config: объект конфигурации;
            - device: устройство для вычислений.
        """
        self.config = config or Config()
        if model_path is None:
            model_path = self.config.get_model_path("binary_classifier")
        else:
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = self.config.base_dir / model_path
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        (self.model, self.class_names) = self._load_model(model_path)
        self.transform = self._create_transform()

    def _load_model(self, model_path: Path) -> tuple:
        """Загружает модель EfficientNet-b0 из указанного пути.

        Аргументы:
            - model_path: путь к файлу модели.

        Возвращает:
            кортеж из загруженной модели и списка имён классов.
        """
        checkpoint = torch.load(
            str(model_path), map_location=self.device, weights_only=False
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            class_names = checkpoint.get("class_names", self.config.binary_classifier_classes)
        else:
            # Старый формат: чекпоинт содержит state_dict напрямую
            state_dict = checkpoint
            class_names = self.config.binary_classifier_classes
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), nn.Linear(num_features, len(class_names))
        )
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return (model, class_names)

    def _create_transform(self) -> transforms.Compose:
        """Создает последовательность преобразований для входных изображений.

        Возвращает:
            объект Compose с трансформациями.
        """
        return transforms.Compose(
            [
                transforms.Resize(
                    (
                        self.config.classifier_image_size,
                        self.config.classifier_image_size,
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.classifier_normalize_mean,
                    std=self.config.classifier_normalize_std,
                ),
            ]
        )

    def classify(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        return_probabilities: bool = True,
    ) -> ClassificationResult:
        """Классифицирует одиночное изображение.

        Аргументы:
            - image: изображение для классификации;
            - return_probabilities: флаг возврата вероятностей классов.

        Возвращает:
            результат классификации.
        """
        return self.classify_batch([image])[0]

    def classify_batch(
        self, images: List[Union[str, Path, np.ndarray, Image.Image]]
    ) -> List[ClassificationResult]:
        """Выполняет классификацию группы изображений.

        Аргументы:
            - images: список изображений для обработки.

        Возвращает:
            список объектов с результатами классификации для каждого изображения.

        Исключения:
            - TypeError: если тип изображения не поддерживается.
        """
        if not images:
            return []
        tensors = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_image = Image.open(str(img)).convert("RGB")
            elif isinstance(img, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif isinstance(img, Image.Image):
                pil_image = img.convert("RGB")
            else:
                raise TypeError(
                    "Изображение должно быть путём к файлу, numpy массивом или PIL Image"
                )
            tensors.append(self.transform(pil_image))
        batch_tensor = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            output = self.model(batch_tensor)
            probs = torch.softmax(output, dim=1)
            (confidences, predicteds) = torch.max(probs, 1)
        results = []
        for i in range(len(images)):
            predicted_idx = predicteds[i].item()
            predicted_class = self.class_names[predicted_idx]
            confidence_pct = confidences[i].item() * 100
            sailboat_prob = probs[i][1].item() if len(self.class_names) > 1 else 0.0
            not_sailboat_prob = probs[i][0].item() if len(self.class_names) > 0 else 0.0
            results.append(
                ClassificationResult(
                    predicted_class=predicted_class,
                    confidence=confidence_pct,
                    sailboat_probability=sailboat_prob,
                    not_sailboat_probability=not_sailboat_prob,
                )
            )
        return results

