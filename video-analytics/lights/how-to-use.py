import cv2
import numpy as np
from ultralytics import YOLO

# Маппинг классов огней
CLASS_MAP = {
    0: 'white',
    1: 'red',
    2: 'green'
}

def classify_vessel_by_lights(image_path, model_path='best_lights.pt', save_result=False, output_path='result_lights.jpg', conf_thresh=0.5):
    """
    Классифицирует статус судна по навигационным огням.
    
    Args:
        image_path: Путь к изображению.
        model_path: Путь к весам YOLO модели.
        save_result: Если True, сохраняет изображение с разметкой в файл.
        output_path: Путь для сохранения результата.
        conf_thresh: Порог уверенности детекции.
    """
    
    # 1. Загрузка модели и инференс
    model = YOLO(model_path)
    results = model(image_path, conf=conf_thresh)
    result = results[0]
    
    # 2. Извлечение данных о боксах
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
        print("Огни не обнаружены.")
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
            # Если центр по X текущего объекта близок к центру предыдущего
            x_tolerance = 40  # пикселей (можно настроить под ваше разрешение)
            
            if abs(curr['center_x'] - prev['center_x']) < x_tolerance:
                current_group.append(curr)
            else:
                # Началась новая мачта/группа
                groups.append(current_group)
                current_group = [curr]
        
        groups.append(current_group)

    # 4. Загрузка изображения для отрисовки
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return

    final_statuses = []

    for group in groups:
        # Получаем последовательность классов в группе (сверху вниз)
        class_sequence = [d['cls'] for d in group]
        status_name = "Неизвестный сигнал"
        color = (0, 0, 255)  # Красный по умолчанию (BGR)

        # --- ЛОГИКА ПРАВИЛ МППСС (Огни) ---
        
        # NUC: Красный - Красный [1, 1]
        if class_sequence == [1, 1]:
            status_name = "NUC"
            # color = (0, 255, 0)  # Зеленый
            
        # RAM: Красный - Белый - Красный [1, 0, 1]
        elif class_sequence == [1, 0, 1]:
            status_name = "RAM"
            # color = (0, 255, 0)
            
        # Fishing (не трал): Красный - Белый [1, 0]
        elif class_sequence == [1, 0]:
            status_name = "Fishing"
            # color = (0, 255, 0)
            
        # Trawling: Зеленый - Белый [2, 0]
        elif class_sequence == [2, 0]:
            status_name = "Trawling"
            # color = (0, 255, 0)
            
        # CBD: Красный - Красный - Красный [1, 1, 1]
        elif class_sequence == [1, 1, 1]:
            status_name = "CBD"
            # color = (0, 255, 0)

        # Вычисляем общий BBox для группы
        x1_min = min(d['xyxy'][0] for d in group)
        y1_min = min(d['xyxy'][1] for d in group)
        x2_max = max(d['xyxy'][2] for d in group)
        y2_max = max(d['xyxy'][3] for d in group)
        group_bbox = [x1_min, y1_min, x2_max, y2_max]
        
        final_statuses.append({
            'status': status_name,
            'bbox': group_bbox,
            'color': color,
            'sequence': class_sequence
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
        print(f"✅ Результат сохранен в {output_path}")
        
    else:
        # Вывод в консоль
        print(f"\n🔦 Найдено сигнальных комбинаций: {len(final_statuses)}")
        for i, item in enumerate(final_statuses):
            print(f"[{i+1}] Статус: {item['status']}")
            print(f"    Последовательность огней: {item['sequence']}")
            print(f"    Координаты группы: {item['bbox']}")
            print("-" * 40)

# Пример использования
if __name__ == "__main__":
    # Вариант 1: Только консоль
    # classify_vessel_by_lights(
    #     image_path='night_ship.png', 
    #     model_path='detect/lights/weights/best.pt', 
    #     save_result=False
    # )
    
    # Вариант 2: Сохранить картинку с разметкой
    classify_vessel_by_lights(
        image_path='generated.png', 
        model_path='detect/train/weights/best.pt', 
        save_result=True, 
        output_path='classified_lights.png'
    )
