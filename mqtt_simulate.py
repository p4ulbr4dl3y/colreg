import json
import time
import paho.mqtt.client as mqtt

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_COMMAND = "colreg/vision/command"
MQTT_TOPIC_RESULT = "colreg/vision/result"

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("✅ Подключено к MQTT брокеру.")
        client.subscribe(MQTT_TOPIC_RESULT)
        
        # Отправляем тестовую команду
        test_msg = {
           "request_id": "test-req-001",
           "action": "analyze",
           "source": "test_images/day/sailing.jpg",
           "is_night": False
        }
        print(f"📤 Отправка команды радара: {json.dumps(test_msg)}")
        client.publish(MQTT_TOPIC_COMMAND, json.dumps(test_msg))
    else:
        print(f"❌ Ошибка подключения: {reason_code}")

def on_message(client, userdata, msg):
    print(f"\n📥 ПОЛУЧЕН ОТВЕТ ОТ НЕЙРОСЕТИ (Топик: {msg.topic})")
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Raw payload: {msg.payload}")
    
    print("\n✅ Симуляция успешно завершена!")
    client.disconnect()

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "test_radar_system")
client.on_connect = on_connect
client.on_message = on_message

print("Подключение к брокеру...")
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Запускаем цикл ожидания ответа
client.loop_forever()
