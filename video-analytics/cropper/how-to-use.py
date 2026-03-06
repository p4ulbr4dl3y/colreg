from ultralytics import YOLO
import cv2
import os
from datetime import datetime

def crop_and_save_boats(image_path, model_name='yolo11n.pt', class_id=8, output_dir='cropped_boats', conf_thresh=0.5):
    """
    Детектирует лодки (класс 8), кропает их и сохраняет как отдельные изображения.
    
    Args:
        image_path (str): Путь к исходному изображению.
        model_name (str): Имя модели YOLO11.
        class_id (int): ID класса для фильтрации (8 = boat в COCO).
        output_dir (str): Папка для сохранения кропов.
        conf_thresh (float): Порог уверенности детекции.
    """
    
    # 1. Создаём папку для сохранения (если не существует)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Загрузка модели
    model = YOLO(model_name)
    
    # 3. Предикт
    results = model(image_path, conf=conf_thresh)
    result = results[0]
    
    # 4. Загрузка изображения для кропа
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Ошибка: не удалось загрузить изображение '{image_path}'")
        return

    boats_found = False
    
    # 5. Обработка детекций
    if result.boxes is not None:
        boxes = result.boxes
        
        for i in range(len(boxes)):
            det_class = int(boxes.cls[i])
            
            # Фильтр: обрабатываем только класс 8 (boat)
            if det_class == class_id:
                boats_found = True
                conf = float(boxes.conf[i])
                
                # Получаем координаты bbox (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                
                # Делаем кроп (важно: y1:y2, x1:x2 для numpy)
                crop = image[y1:y2, x1:x2]
                
                # Формируем имя файла с уверенностью и timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"boat_{i}_{conf:.2f}_{timestamp}.jpg"
                save_path = os.path.join(output_dir, filename)
                
                # Сохраняем кроп
                cv2.imwrite(save_path, crop)
                
                print(f"✅ Лодка #{i+1} сохранена: {save_path}")
                print(f"   BBox: [{x1}, {y1}, {x2}, {y2}], Conf: {conf:.2f}")
                print(f"   Размер кропа: {crop.shape[1]}x{crop.shape[0]}px")

    # 6. Итог
    if boats_found:
        print(f"\n💾 Всего сохранено лодок: {sum(1 for b in result.boxes if int(b.cls) == class_id)}")
        print(f"📁 Папка сохранения: '{output_dir}'")
    else:
        print(f"\n⚠️ Лодки (класс {class_id}) не обнаружены на изображении.")

# Пример запуска
if __name__ == "__main__":
    img_path = 'image.png'
    
    if os.path.exists(img_path):
        crop_and_save_boats(
            image_path=img_path,
            model_name='yolo11n.pt',
            class_id=8,
            output_dir='cropped_boats',  # Папка для кропов
            conf_thresh=0.5
        )
    else:
        print(f"❌ Файл '{img_path}' не найден.")
