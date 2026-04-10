import json
import logging
import time
from typing import Any, Dict

import cv2
import paho.mqtt.client as mqtt

from pipeline import VideoAnalyticsPipeline

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ColregVisionNode")

# Конфигурация MQTT (в продакшене эти параметры должны браться из переменных окружения)
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_COMMAND = (
    "colreg/vision/command"  # Топик, куда радар/система шлет команды на анализ
)
MQTT_TOPIC_RESULT = (
    "colreg/vision/result"  # Топик, куда мы отдаем результаты классификации
)


class VisionNode:
    """
    MQTT-адаптер для конвейера видеоаналитики COLREG.
    Слушает команды на анализ, захватывает кадры с камеры (или читает файлы)
    и публикует результаты обратно в MQTT шину.
    """

    def __init__(self):
        logger.info("Initializing Video Analytics Pipeline...")
        self.pipeline = VideoAnalyticsPipeline()
        # "Прогрев" моделей (первый инференс всегда долгий, сделаем его на пустой картинке)
        dummy_img = cv2.imread("test_images/day/sea.webp")
        if dummy_img is not None:
            self.pipeline.process(dummy_img)
        logger.info("Pipeline initialized and models warmed up.")

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, "colreg_vision_node"
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: dict,
        reason_code: int,
        properties: Any,
    ) -> None:
        """Колбэк при успешном подключении к MQTT брокеру."""
        if reason_code == 0:
            logger.info(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
            # Подписываемся на команды
            client.subscribe(MQTT_TOPIC_COMMAND)
            logger.info(f"Subscribed to topic: {MQTT_TOPIC_COMMAND}")
        else:
            logger.error(
                f"Failed to connect to MQTT broker. Reason code: {reason_code}"
            )

    def _on_message(
        self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage
    ) -> None:
        """Колбэк при получении сообщения из топика."""
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            logger.info(f"Received command: {payload}")

            # Ожидаемый формат payload:
            # {
            #    "request_id": "req-123",
            #    "action": "analyze",
            #    "source": "rtsp://camera_ip/stream" или "/path/to/image.jpg",
            #    "is_night": false
            # }

            action = payload.get("action")
            if action == "analyze":
                self._handle_analyze_command(payload)
            else:
                logger.warning(f"Unknown action: {action}")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON payload: {msg.payload}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _handle_analyze_command(self, payload: Dict[str, Any]) -> None:
        """Обрабатывает команду на анализ видео/изображения."""
        request_id = payload.get("request_id", "unknown")
        source = payload.get("source")
        is_night = payload.get("is_night", False)

        if not source:
            self._publish_error(request_id, "Missing 'source' in command payload.")
            return

        logger.info(f"[{request_id}] Starting analysis on source: {source}")
        start_time = time.time()

        try:
            # 1. Получаем кадр
            # Если это RTSP поток, мы бы использовали cv2.VideoCapture(source)
            # Для простоты примера предполагаем, что source - это путь к файлу
            image = cv2.imread(source)

            if image is None:
                self._publish_error(
                    request_id, f"Failed to load image from source: {source}"
                )
                return

            # 2. Прогоняем через пайплайн
            result = self.pipeline.process(image, is_night=is_night)

            # 3. Формируем ответ (Сериализация PipelineResult в JSON)
            boats_data = []
            for boat in result.boats:
                boats_data.append(
                    {
                        "boat_id": boat.boat_id,
                        "bbox": boat.bbox,
                        "vessel_type": boat.final_vessel_type,
                        "confidence": round(boat.final_vessel_type_confidence, 2),
                    }
                )

            response = {
                "request_id": request_id,
                "status": "success",
                "is_night": result.is_night,
                "boat_count": result.boat_count,
                "boats": boats_data,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            }

            # 4. Отправляем результат
            self.client.publish(MQTT_TOPIC_RESULT, json.dumps(response))
            logger.info(
                f"[{request_id}] Analysis complete. Found {result.boat_count} boats. Published to {MQTT_TOPIC_RESULT}"
            )

        except Exception as e:
            logger.error(f"[{request_id}] Analysis failed: {e}")
            self._publish_error(request_id, str(e))

    def _publish_error(self, request_id: str, error_message: str) -> None:
        """Публикует сообщение об ошибке."""
        response = {
            "request_id": request_id,
            "status": "error",
            "message": error_message,
        }
        self.client.publish(MQTT_TOPIC_RESULT, json.dumps(response))

    def start(self):
        """Запускает MQTT клиента в блокирующем цикле."""
        try:
            logger.info("Connecting to MQTT broker...")
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_forever()
        except KeyboardInterrupt:
            logger.info("Disconnecting from MQTT broker...")
            self.client.disconnect()
        except Exception as e:
            logger.error(f"Failed to start MQTT node: {e}")


if __name__ == "__main__":
    node = VisionNode()
    node.start()
