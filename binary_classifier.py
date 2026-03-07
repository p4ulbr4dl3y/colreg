"""
Binary classifier module for sailboat detection.

Classifies boat crops as sailboat or not sailboat.
Works independently of the pipeline - accepts any image.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from config import Config


@dataclass
class ClassificationResult:
    """Result of binary classification."""

    predicted_class: str  # 'sailboat' or 'not_sailboat'
    confidence: float  # Confidence percentage (0-100)
    sailboat_probability: float  # Probability of being sailboat (0-1)
    not_sailboat_probability: float  # Probability of not being sailboat (0-1)

    @property
    def is_sailboat(self) -> bool:
        return self.predicted_class == "sailboat"


class BinaryClassifier:
    """
    Binary classifier for sailboat detection.

    This class is pipeline-agnostic - load once, use multiple times
    on any image without pipeline dependencies.

    Example:
        >>> classifier = BinaryClassifier()
        >>> result = classifier.classify('boat_crop.jpg')
        >>> if result.is_sailboat:
        ...     print(f"Sailboat detected: {result.confidence:.1f}%")
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[Config] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the binary classifier.

        Args:
            model_path: Path to model weights (.pth file).
            config: Configuration object. Uses default if None.
            device: Device to run inference on ('cuda', 'cpu', or None for auto).
        """
        self.config = config or Config()

        # Resolve model path
        if model_path is None:
            model_path = self.config.get_model_path("binary_classifier")
        else:
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = self.config.base_dir / model_path

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        self.model, self.class_names = self._load_model(model_path)

        # Create transform
        self.transform = self._create_transform()

    def _load_model(self, model_path: Path) -> tuple:
        """Load trained EfficientNet model from checkpoint."""
        checkpoint = torch.load(
            str(model_path), map_location=self.device, weights_only=False
        )

        # Map EfficientNet versions
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

        version = checkpoint.get("efficientnet_version", "b0")
        model = model_variants[version](weights=None)

        # Replace classifier head
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), nn.Linear(num_features, 2)
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        class_names = checkpoint.get("class_names", ["not_sailboat", "sailboat"])

        return model, class_names

    def _create_transform(self) -> transforms.Compose:
        """Create image transformation pipeline."""
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
        """
        Classify a single image.

        Args:
            image: Input image as file path, numpy array (BGR), or PIL Image (RGB).
            return_probabilities: Whether to include all class probabilities.

        Returns:
            ClassificationResult with predicted class and confidence.

        Example:
            >>> crop = cv2.imread('boat_crop.jpg')
            >>> result = classifier.classify(crop)
            >>> print(f"{result.predicted_class}: {result.confidence:.1f}%")
        """
        # Load and convert image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Convert BGR (OpenCV) to RGB (PIL)
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError("Image must be file path, numpy array, or PIL Image")

        # Apply transforms
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        # Extract results
        predicted_idx = predicted.item()
        predicted_class = self.class_names[predicted_idx]
        confidence_pct = confidence.item() * 100

        sailboat_prob = probs[0][1].item() if len(self.class_names) > 1 else 0.0
        not_sailboat_prob = probs[0][0].item() if len(self.class_names) > 0 else 0.0

        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence_pct,
            sailboat_probability=sailboat_prob,
            not_sailboat_probability=not_sailboat_prob,
        )
