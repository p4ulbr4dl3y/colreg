import json

import paho.mqtt.client as mqtt

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_COMMAND = "colreg/vision/command"
MQTT_TOPIC_RESULT = "colreg/vision/result"


def on_connect(client, userdata, flags, reason_code, properties):
    """Обработчик события подключения к MQTT брокеру.

    При успешном подключении подписывается на топик результатов и
    отправляет тестовую команду анализа.
    """
    if reason_code == 0:
        print("Подключено к MQTT брокеру.")
        client.subscribe(MQTT_TOPIC_RESULT)
        test_msg = {
            "request_id": "test-req-001",
            "action": "analyze",
            "source": "test_images/day/sailing.jpg",
            "is_night": False,
        }
        print(f"Отправка команды радара: {json.dumps(test_msg)}")
        client.publish(MQTT_TOPIC_COMMAND, json.dumps(test_msg))
    else:
        print(f"Ошибка подключения: {reason_code}")


def on_message(client, userdata, msg):
    """Обработчик входящих сообщений (ответов нейросети).

    Выводит полученные данные в консоль и отключается от брокера.
    """
    print(f"\nПОЛУЧЕН ОТВЕТ ОТ НЕЙРОСЕТИ (Топик: {msg.topic})")
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception:
        print(f"Сырая полезная нагрузка: {msg.payload}")
    print("\nСимуляция успешно завершена!")
    client.disconnect()


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "test_radar_system")
client.on_connect = on_connect
client.on_message = on_message
print("Подключение к брокеру...")
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()
