from ultralytics import YOLO
from IPython.display import Image, display
import cv2

# 1. Загрузка модели
# Вы можете использовать предобученную модель ('yolov8n.pt', 'yolov8s.pt', etc.)
# или путь к своим весам ('runs/detect/train/weights/best.pt')
model = YOLO('detect/train3/weights/best.pt')
# model = YOLO('yolo11n.pt')

# 2. Подготовка источника данных
# Замените путь на ваше изображение, видео или используйте 0 для веб-камеры
source = 'https://maritime-executive.com/media/images/article/UK-MOD-Yantar-infrared.c90c74.png' # Пример изображения из интернета
# source = 'path/to/your/image.jpg'               # Локальный файл
# source = 0                                      # Веб-камера

# 3. Проведение инференса
# save=True сохраняет результат во временную папку runs/detect/predict
# show=False предотвращает открытие отдельного окна OpenCV (важно для Jupyter)
results = model.predict(source=source, save=True, show=False, conf=0.25)

# 4. Отображение результата в ячейке Jupyter
# Результат сохраняется в runs/detect/predict/<имя_файла>.jpg
# Берем путь к сохраненному файлу из объекта results
result_path = results[0].save_dir + "/" + results[0].names.get(int(results[0].boxes.cls[0]), "result") if len(results[0].boxes) > 0 else None

# Более надежный способ получить путь к сохраненному изображению:
# Библиотека ultralytics автоматически сохраняет картинки, если указан флаг save=True
# Путь обычно выглядит так: runs/detect/predict/bus.jpg
import os
saved_images = []
for r in results:
    # Путь сохранения формируется автоматически
    save_path = r.save_dir
    # Находим все jpg файлы в папке сохранения
    files = [f for f in os.listdir(save_path) if f.endswith('.jpg')]
    if files:
        full_path = os.path.join(save_path, files[0])
        saved_images.append(full_path)

# Вывод изображений
if saved_images:
    print(f"Обнаружено объектов: {len(results[0].boxes)}")
    for img_path in saved_images:
        display(Image(filename=img_path))
else:
    print("Изображение не было сохранено или объекты не найдены.")
