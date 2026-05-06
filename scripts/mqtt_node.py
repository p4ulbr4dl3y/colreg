import json
import logging
import time
from typing import Any, Dict

import cv2
import paho.mqtt.client as mqtt

from colreg_vision.pipeline import VideoAnalyticsPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ColregVisionNode")
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_COMMAND = "colreg/vision/command"
MQTT_TOPIC_RESULT = "colreg/vision/result"


class VisionNode:
    """Узел для обработки команд анализа видео через MQTT.

    Класс управляет жизненным циклом конвейера видеоаналитики и обеспечивает взаимодействие с MQTT брокером: подписку на команды и публикацию результатов.

    Атрибуты:
        - pipeline: экземпляр конвейера видеоаналитики;
        - client: клиент для работы с MQTT брокером.
    """

    def __init__(self):
        logger.info("Инициализация конвейера видеоаналитики...")
        self.pipeline = VideoAnalyticsPipeline()
        dummy_img = cv2.imread("test_images/day/sea.webp")
        if dummy_img is not None:
            self.pipeline.process(dummy_img)
        logger.info("Конвейер инициализирован, модели прогреты.")
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
        """Обработчик события успешного подключения к MQTT брокеру.

        Оформляет подписку на топик команд после установки соединения.
        """
        if reason_code == 0:
            logger.info(
                f"Успешное подключение к MQTT брокеру на {MQTT_BROKER}:{MQTT_PORT}"
            )
            client.subscribe(MQTT_TOPIC_COMMAND)
            logger.info(f"Подписка на топик оформлена: {MQTT_TOPIC_COMMAND}")
        else:
            logger.error(
                f"Не удалось подключиться к MQTT брокеру. Код ошибки: {reason_code}"
            )

    def _on_message(
        self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage
    ) -> None:
        """Обработчик входящих сообщений MQTT.

        Парсит JSON полезную нагрузку и вызывает соответствующий метод обработки.
        """
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            logger.info(f"Получена команда: {payload}")
            action = payload.get("action")
            if action == "analyze":
                self._handle_analyze_command(payload)
            else:
                logger.warning(f"Неизвестное действие: {action}")
        except json.JSONDecodeError:
            logger.error(f"Не удалось распарсить JSON полезной нагрузки: {msg.payload}")
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {e}")

    def _handle_analyze_command(self, payload: Dict[str, Any]) -> None:
        """Обрабатывает команду анализа видео.

        Загружает изображение, вызывает конвейер видеоаналитики и публикует
        результат в топик результатов.
        """
        request_id = payload.get("request_id", "unknown")
        source = payload.get("source")
        is_night = payload.get("is_night", False)
        if not source:
            self._publish_error(
                request_id, "Отсутствует 'source' в полезной нагрузке команды."
            )
            return
        logger.info(f"[{request_id}] Запуск анализа источника: {source}")
        start_time = time.time()
        try:
            image = cv2.imread(source)
            if image is None:
                self._publish_error(
                    request_id,
                    f"Не удалось загрузить изображение из источника: {source}",
                )
                return
            result = self.pipeline.process(image, is_night=is_night)
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
            self.client.publish(MQTT_TOPIC_RESULT, json.dumps(response))
            logger.info(
                f"[{request_id}] Анализ завершен. Найдено судов: {result.boat_count}. Опубликовано в {MQTT_TOPIC_RESULT}"
            )
        except Exception as e:
            logger.error(f"[{request_id}] Ошибка анализа: {e}")
            self._publish_error(request_id, str(e))

    def _publish_error(self, request_id: str, error_message: str) -> None:
        """Публикует сообщение об ошибке в топик результатов."""
        response = {
            "request_id": request_id,
            "status": "error",
            "message": error_message,
        }
        self.client.publish(MQTT_TOPIC_RESULT, json.dumps(response))

    def start(self):
        """Запускает MQTT узел и переходит в цикл ожидания команд."""
        try:
            logger.info("Подключение к MQTT брокеру...")
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_forever()
        except KeyboardInterrupt:
            logger.info("Отключение от MQTT брокера...")
            self.client.disconnect()
        except Exception as e:
            logger.error(f"Не удалось запустить MQTT узел: {e}")


if __name__ == "__main__":
    node = VisionNode()
    node.start()
