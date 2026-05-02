"""
Модуль бинарной классификации для обнаружения парусных судов.

Классифицирует вырезанные изображения судов как парусные или непарусные.
Работает независимо от конвейера — принимает любое изображение.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from colreg_vision.core.config import Config


@dataclass
class ClassificationResult:
    """Результат бинарной классификации."""

    predicted_class: str  # 'sailboat' или 'not_sailboat'
    confidence: float  # Уверенность в процентах (0-100)
    sailboat_probability: float  # Вероятность быть парусным (0-1)
    not_sailboat_probability: float  # Вероятность быть непарусным (0-1)

    @property
    def is_sailboat(self) -> bool:
        return self.predicted_class == "sailboat"


class BinaryClassifier:
    """
    Бинарный классификатор для обнаружения парусных судов.

    Этот класс не зависит от конвейера — загрузите один раз, используйте многократно
    на любых изображениях без зависимостей от конвейера.

    Пример:
        >>> classifier = BinaryClassifier()
        >>> result = classifier.classify('boat_crop.jpg')
        >>> if result.is_sailboat:
        ...     print(f"Обнаружено парусное судно: {result.confidence:.1f}%")
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[Config] = None,
        device: Optional[str] = None,
    ):
        """
        Инициализировать бинарный классификатор.

        Args:
            model_path: Путь к весам модели (.pth файл).
            config: Объект конфигурации. Используется по умолчанию, если None.
            device: Устройство для инференса ('cuda', 'cpu' или None для авто).
        """
        self.config = config or Config()

        # Разрешить путь к модели
        if model_path is None:
            model_path = self.config.get_model_path("binary_classifier")
        else:
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = self.config.base_dir / model_path

        # Установить устройство
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Загрузить модель
        self.model, self.class_names = self._load_model(model_path)

        # Создать трансформацию
        self.transform = self._create_transform()

    def _load_model(self, model_path: Path) -> tuple:
        """Загрузить обученную модель EfficientNet из контрольной точки."""
        checkpoint = torch.load(
            str(model_path), map_location=self.device, weights_only=False
        )

        # Сопоставить версии EfficientNet
        model_variants = {
            "b0": models.efficientnet_b0,
            "b1": models.efficientnet_b1,
            "b2": models.efficientnet_b2,
            "b3": models.efficientnet_b3,
            "b4": models.efficientnet_b4,
            "b5": models.efficientnet_b5,
            "b6": models.efficientnet_b6,
            "b7": models.efficientnet_b7,
        }

        # Если чекпоинт - это OrderedDict (только веса), используем b0 по умолчанию
        if (
            isinstance(checkpoint, (dict, torch.nn.modules.container.Sequential))
            and "model_state_dict" not in checkpoint
        ):
            state_dict = checkpoint
            version = "b0"
            class_names = ["not_sailboat", "sailboat"]
        else:
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            version = checkpoint.get("efficientnet_version", "b0")
            class_names = checkpoint.get("class_names", ["not_sailboat", "sailboat"])

        model = model_variants[version](weights=None)

        # Заменить головку классификатора
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), nn.Linear(num_features, len(class_names))
        )

        # Загрузить веса
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        return model, class_names

    def _create_transform(self) -> transforms.Compose:
        """Создать конвейер трансформаций изображения."""
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
        """Классифицировать одно изображение (прокси для classify_batch)."""
        return self.classify_batch([image])[0]

    def classify_batch(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]],
    ) -> List[ClassificationResult]:
        """
        Классифицировать список изображений за один проход (Batch Inference).

        Args:
            images: Список входных изображений.

        Returns:
            Список ClassificationResult.
        """
        if not images:
            return []

        tensors = []
        for img in images:
            # Загрузить и конвертировать изображение
            if isinstance(img, (str, Path)):
                pil_image = Image.open(str(img)).convert("RGB")
            elif isinstance(img, np.ndarray):
                # Конвертировать BGR (OpenCV) в RGB (PIL)
                pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif isinstance(img, Image.Image):
                pil_image = img.convert("RGB")
            else:
                raise TypeError(
                    "Изображение должно быть путём к файлу, numpy массивом или PIL Image"
                )

            # Применить трансформации и добавить в список
            tensors.append(self.transform(pil_image))

        # Собрать батч и отправить на устройство
        batch_tensor = torch.stack(tensors).to(self.device)

        # Запустить инференс
        with torch.no_grad():
            output = self.model(batch_tensor)
            probs = torch.softmax(output, dim=1)
            confidences, predicteds = torch.max(probs, 1)

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
