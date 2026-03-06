#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ==================== КОНФИГУРАЦИЯ ====================

MODEL_PATH = Path("best_model.pth")
EFFICIENTNET_VERSION = 'b0'  # Должна совпадать с версией при обучении
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== ЗАГРУЗКА МОДЕЛИ ====================

def load_model(model_path):
    """Загружает обученную модель."""

    print(f"🔥 Устройство: {DEVICE}")
    print(f"📦 Загрузка модели из: {model_path}")

    # Загружаем чекпоинт
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # Маппинг версий EfficientNet
    model_variants = {
        'b0': models.efficientnet_b0,
        'b1': models.efficientnet_b1,
        'b2': models.efficientnet_b2,
        'b3': models.efficientnet_b3,
        'b4': models.efficientnet_b4,
        'b5': models.efficientnet_b5,
        'b6': models.efficientnet_b6,
        'b7': models.efficientnet_b7,
    }

    # Создаём модель
    model = model_variants[checkpoint['efficientnet_version']](weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, 2)  # Бинарная классификация
    )

    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    class_names = checkpoint['class_names']
    print(f"✅ Модель загружена!")
    print(f"📁 Классы: {class_names}")
    print(f"🏆 Accuracy на валидации: {checkpoint['valid_acc']:.4f}")

    return model, class_names

# ==================== ПРЕДОБРАБОТКА ====================

def get_transform():
    """Трансформы для инференса (должны совпадать с валидацией)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ==================== ИНФЕРЕНС ====================

def predict_image(model, image_path, class_names, show_image=True):
    """
    Предсказание для одного изображения.

    Args:
        model: Обученная модель
        image_path: Путь к изображению
        class_names: Список классов из чекпоинта
        show_image: Показать изображение с предсказанием

    Returns:
        dict с результатом
    """

    transform = get_transform()

    # Загрузка изображения
    img = Image.open(image_path).convert('RGB')
    original_size = img.size

    # Предобработка
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Предсказание
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    # Результаты
    result = {
        'image_path': str(image_path),
        'predicted_class': class_names[predicted.item()],
        'confidence': confidence.item() * 100,
        'all_probabilities': {
            class_names[i]: probs[0][i].item() * 100
            for i in range(len(class_names))
        }
    }

    # Вывод
    print("\n" + "="*60)
    print(f"🖼️  Изображение: {image_path}")
    print(f"   Размер: {original_size[0]}x{original_size[1]}")
    print("="*60)
    print(f"🎯 Предсказание: {result['predicted_class']}")
    print(f"📊 Уверенность: {result['confidence']:.2f}%")
    print("\n📈 Все вероятности:")
    for cls, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(prob / 5)
        print(f"   {cls:15s}: {prob:6.2f}% {bar}")

    # Визуализация
    if show_image:
        plt.figure(figsize=(10, 5))

        # Изображение
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')

        # Бар chart вероятностей
        plt.subplot(1, 2, 2)
        classes = list(result['all_probabilities'].keys())
        probs = list(result['all_probabilities'].values())
        colors = ['#2ecc71' if c == result['predicted_class'] else '#95a5a6' for c in classes]
        plt.barh(classes, probs, color=colors)
        plt.xlabel('Probability (%)')
        plt.title('Class Probabilities')
        plt.xlim(0, 100)
        for i, v in enumerate(probs):
            plt.text(v + 1, i, f'{v:.1f}%', va='center')

        plt.tight_layout()
        plt.savefig(f"./prediction_{Path(image_path).stem}.png", dpi=150)
        print(f"\n📊 Визуализация сохранена: ./prediction_{Path(image_path).stem}.png")
        plt.show()

    return result

def predict_batch(model, image_folder, class_names, extensions=None):
    """
    Предсказание для всех изображений в папке.

    Args:
        model: Обученная модель
        image_folder: Путь к папке с изображениями
        class_names: Список классов
        extensions: Список расширений файлов

    Returns:
        list с результатами
    """

    if extensions is None:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

    folder_path = Path(image_folder)
    if not folder_path.exists():
        print(f"❌ Папка не найдена: {folder_path}")
        return []

    # Сбор всех изображений
    image_files = []
    for ext in extensions:
        image_files.extend(folder_path.glob(ext))

    if not image_files:
        print(f"❌ Изображения не найдены в: {folder_path}")
        return []

    print(f"\n📁 Найдено изображений: {len(image_files)}")
    print("="*60)

    results = []
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]")
        result = predict_image(model, img_path, class_names, show_image=False)
        results.append(result)

    # Статистика
    print("\n" + "="*60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("="*60)

    sailboat_count = sum(1 for r in results if r['predicted_class'] == 'sailboat')
    not_sailboat_count = len(results) - sailboat_count

    print(f"Всего изображений: {len(results)}")
    print(f"🚢 sailboat: {sailboat_count} ({sailboat_count/len(results)*100:.1f}%)")
    print(f"⛵ not_sailboat: {not_sailboat_count} ({not_sailboat_count/len(results)*100:.1f}%)")

    # Сохранение результатов
    import json
    with open('./predictions.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Результаты сохранены: ./predictions.json")

    return results

# ==================== MAIN ====================

def main():
    # Загрузка модели
    model, class_names = load_model(MODEL_PATH)

    print("\n" + "="*60)
    print("🔮 ИНФЕРЕНС - ВЫБЕРИТЕ РЕЖИМ")
    print("="*60)
    print("1. Предсказание для одного изображения")
    print("2. Предсказание для всех изображений в папке")
    print("="*60)

    # Для демонстрации - раскомментируйте нужный вариант

    # ─── ВАРИАНТ 1: ОДНО ИЗОБРАЖЕНИЕ ──────────────────────────
    image_path = input("\n📷 Введите путь к изображению: ").strip()
    if Path(image_path).exists():
        predict_image(model, image_path, class_names, show_image=True)
    else:
        print(f"❌ Файл не найден: {image_path}")

    # ─── ВАРИАНТ 2: ПАКЕТНЫЙ ИНФЕРЕНС ─────────────────────────
    # folder_path = input("\n📁 Введите путь к папке с изображениями: ").strip()
    # predict_batch(model, folder_path, class_names)

    print("\n✅ Готово!")

if __name__ == "__main__":
    main()
