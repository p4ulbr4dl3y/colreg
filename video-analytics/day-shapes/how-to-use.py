import cv2
import numpy as np
from ultralytics import YOLO

# Маппинг классов из вашей модели
CLASS_MAP = {
    0: 'ball',
    1: 'cone_up',
    2: 'cone_down',
    3: 'diamond',
    4: 'cylinder'
}

def classify_vessel_status(image_path, model_path='best.pt', save_result=False, output_path='result.jpg'):
    """
    Классифицирует статус судна по дневным знакам.
    
    Args:
        image_path: Путь к изображению.
        model_path: Путь к весам YOLO модели.
        save_result: Если True, сохраняет изображение с разметкой в файл.
        output_path: Путь для сохранения результата.
    """
    
    # 1. Загрузка модели и инференс
    model = YOLO(model_path)
    results = model(image_path, conf=0.5) # conf threshold можно настроить
    result = results[0]
    
    # 2. Извлечение данных о боксах
    # Формат данных: [class_id, x_center, y_center, x1, y1, x2, y2]
    detections = []
    if result.boxes is not None:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls_id = int(box.cls[0])
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            detections.append({
                'cls': cls_id,
                'name': CLASS_MAP.get(cls_id, 'unknown'),
                'xyxy': [x1, y1, x2, y2],
                'center_x': center_x,
                'center_y': center_y
            })

    # Если ничего не найдено
    if not detections:
        print("Знаки не обнаружены.")
        return

    # 3. Группировка по вертикали (по мачтам)
    # Сортируем все детекции сверху вниз (по Y центру)
    detections.sort(key=lambda x: x['center_y'])
    
    groups = []
    current_group = []
    
    if detections:
        current_group.append(detections[0])
        
        for i in range(1, len(detections)):
            prev = current_group[-1]
            curr = detections[i]
            
            # Логика группировки:
            # Если центр по X текущего объекта близок к центру предыдущего (допуск 40 пикселей),
            # считаем, что они на одной мачте.
            x_tolerance = 40 
            
            if abs(curr['center_x'] - prev['center_x']) < x_tolerance:
                current_group.append(curr)
            else:
                # Началась новая мачта/группа
                groups.append(current_group)
                current_group = [curr]
        
        groups.append(current_group)

    # 4. Классификация статусов и подготовка вывода
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return

    final_statuses = []

    for group in groups:
        # Получаем последовательность классов в группе (сверху вниз)
        class_sequence = [d['cls'] for d in group]
        status_name = "Неизвестный сигнал"
        color = (0, 0, 255) # Красный по умолчанию

        # --- ЛОГИКА ПРАВИЛ МППСС ---
        
        # NUC: Шар - Шар [0, 0]
        if class_sequence == [0, 0]:
            status_name = "NUC"
            # color = (0, 255, 0) # Зеленый
            
        # RAM: Шар - Ромб - Шар [0, 3, 0]
        elif class_sequence == [0, 3, 0]:
            status_name = "RAM"
            # color = (0, 255, 0)
            
        # CBD: Цилиндр [4]
        elif class_sequence == [4]:
            status_name = "CBD"
            # color = (0, 255, 0)
            
        # Fishing / Trawling: Конусы вершинами друг к другу
        # Визуально: сверху конус острием вниз (2), снизу острием вверх (1)
        # Последовательность: [2, 1]
        elif class_sequence == [2, 1]: 
            status_name = "Fishing / Trawling"
            # color = (0, 255, 0)
            
        # (Опционально) Если модель путает порядок конусов, можно добавить проверку [1, 2]
        # elif class_sequence == [1, 2]: ...

        # Вычисляем общий BBox для группы
        x1_min = min(d['xyxy'][0] for d in group)
        y1_min = min(d['xyxy'][1] for d in group)
        x2_max = max(d['xyxy'][2] for d in group)
        y2_max = max(d['xyxy'][3] for d in group)
        group_bbox = [x1_min, y1_min, x2_max, y2_max]
        
        final_statuses.append({
            'status': status_name,
            'bbox': group_bbox,
            'color': color
        })

    # 5. Вывод результатов
    if save_result:
        # Рисуем только общие боксы групп
        for item in final_statuses:
            x1, y1, x2, y2 = map(int, item['bbox'])
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), item['color'], 3)
            
            # Подпись
            label = f"{item['status']}"
            cv2.putText(image_cv, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, item['color'], 2)
        
        cv2.imwrite(output_path, image_cv)
        print(f"Результат сохранен в {output_path}")
        
    else:
        # Вывод в консоль
        print(f"\nНайдено сигнальных комбинаций: {len(final_statuses)}")
        for i, item in enumerate(final_statuses):
            print(f"[{i+1}] Статус: {item['status']}")
            print(f"    Координаты группы: {item['bbox']}")
            print("-" * 30)

# Пример использования
if __name__ == "__main__":
    # Замените 'best.pt' на путь к вашему файлу весов
    # Замените 'ship_image.jpg' на путь к тестовому изображению
    
    # Вариант 1: Только консоль
    classify_vessel_status(image_path='image.png', model_path='detect/train/weights/best.pt', save_result=True, output_path='classified_ship.png')
    
    # Вариант 2: Сохранить картинку с разметкой
    # classify_vessel_status(image_path='ship_image.jpg', model_path='best.pt', save_result=True, output_path='classified_ship.jpg')
