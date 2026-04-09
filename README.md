# COLREG — Видеоаналитика для морской навигации

Конвейер видеоаналитики для автоматической классификации судов согласно **МППСС-72** (Международные правила предупреждения столкновений судов в море).

## Возможности

- 🚤 **Обнаружение судов** — детектирование судов на изображениях с помощью YOLO
- ⛵ **Бинарная классификация** — определение типа: парусное или механическое
- 🌙 **Ночной режим** — инфракрасное обнаружение объектов
- 🔷 **Дневные фигуры** — классификация по дневным сигналам (шары, конусы, ромбы, цилиндры)
- 🚦 **Навигационные огни** — классификация по ночным сигналам (красный, белый, зелёный)

## Типы судов МППСС-72

Система распознаёт следующие типы судов:

| Тип | Описание | Сигнал (дневной) | Сигнал (ночной) |
|-----|----------|------------------|-----------------|
| **NUC** | Не может управляться | 2 шара | 2 красных огня |
| **RAM** | Ограничено в возможности маневрировать | шар-ромб-шар | красный-белый-красный |
| **CBD** | Стеснено своей осадкой | цилиндр | 3 красных огня |
| **FISHING** | Занято ловом рыбы | 2 конуса вершинами вместе | красный-белый |
| **SAIL** | Парусное судно | — | — |
| **MECH** | Механическое судно | — | — |

## Установка

```bash
pip install -r requirements.txt
```

### Зависимости

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO
- Pillow
- NumPy

## Быстрый старт

### Дневной режим

```python
import cv2
from pipeline import VideoAnalyticsPipeline, draw_results

# Инициализация конвейера
pipeline = VideoAnalyticsPipeline()

# Загрузка изображения
image = cv2.imread('frame.jpg')

# Анализ (дневной режим)
result = pipeline.process(image, is_night=False)

# Визуализация результатов
vis = draw_results(image, result)
cv2.imwrite('output.png', vis)

# Вывод информации
print(f"Обнаружено судов: {result.boat_count}")
for boat in result.boats:
    print(f"Судно {boat.boat_id}: тип = {boat.final_vessel_type}")
```

### Ночной режим

```python
import cv2
from pipeline import VideoAnalyticsPipeline, draw_results

pipeline = VideoAnalyticsPipeline()

# Загрузка ИК и видимого изображений
ir_image = cv2.imread('thermal_frame.png')
visible_image = cv2.imread('visible_frame.png')

# Анализ (ночной режим)
result = pipeline.process_night(ir_image, visible_image)

# Визуализация
vis = draw_results(ir_image, result)
cv2.imwrite('output_night.png', vis)
```

## Модульное использование

Все функции конвейера доступны отдельно для автономного использования:

### Обнаружение и кадрирование судов

```python
from boat_detector import detect_and_crop_boats

image = cv2.imread('scene.jpg')
detections = detect_and_crop_boats(image)

for det in detections:
    print(f"Судно: уверенность = {det.confidence:.2f}, размер = {det.width}x{det.height}")
```

### Бинарная классификация

```python
from binary_classifier import BinaryClassifier

classifier = BinaryClassifier()
result = classifier.classify('boat_crop.jpg')

print(f"Тип: {result.predicted_class}, уверенность: {result.confidence:.1f}%")
```

### Классификация дневных фигур

```python
from day_shapes import classify_day_shapes

image = cv2.imread('vessel.png')
types = classify_day_shapes(image)

for vtype in types:
    print(f"Тип судна: {vtype.vessel_type}")
```

### Классификация навигационных огней

```python
from lights import classify_lights

image = cv2.imread('night_vessel.png')
types = classify_lights(image)

for vtype in types:
    print(f"Тип судна: {vtype.vessel_type}")
```

### Инфракрасное обнаружение

```python
from infrared_detector import detect_infrared_objects

image = cv2.imread('thermal_scene.png')
detections = detect_infrared_objects(image)

for det in detections:
    print(f"Объект: {det.class_name}, уверенность: {det.confidence:.2f}")
```

## Конфигурация

Конфигурация определяется в классе `Config`:

```python
from config import Config

config = Config()

# Пути к моделям
config.boat_detector.path = "models/yolo11n.pt"
config.binary_classifier.path = "models/binary-classifier.pth"
config.infrared_detector.path = "models/infrared.pt"
config.day_shapes.path = "models/day-shapes.pt"
config.lights.path = "models/lights.pt"

# Пороги уверенности
config.boat_detector.confidence_threshold = 0.5
config.infrared_detector.confidence_threshold = 0.25
```

## Структура проекта

```
colreg/
├── __init__.py              # Инициализация пакета
├── config.py                # Конфигурация
├── boat_detector.py         # Обнаружение судов (YOLO)
├── binary_classifier.py     # Бинарная классификация (парусное/механическое)
├── infrared_detector.py     # Инфракрасное обнаружение (ночной режим)
├── day_shapes.py            # Классификация дневных фигур
├── lights.py                # Классификация навигационных огней
├── pipeline.py              # Главный конвейер анализа
└── README.md                # Документация
```

## Приоритет классификации

Итоговый тип судна определяется по приоритету (от высшего к низшему):

1. **NUC** (Не может управляться)
2. **RAM** (Ограничено в возможности маневрировать)
3. **CBD** (Стеснено своей осадкой)
4. **FISHING** (Занято ловом рыбы)
5. **SAIL / MECH** (Парусное / Механическое — от бинарного классификатора)

Дневные фигуры и навигационные огни имеют приоритет над бинарной классификацией.
