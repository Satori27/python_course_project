from kafka import KafkaProducer
import json


# Настроим Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='kafka:9092',  # укажите ваш Kafka брокер
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
# Пример сообщения для отправки в Kafka
for i in range(100):
    message = {"user_id": "7cbc7c3b-3b42-4e3c-9908-7bbc661c6659", "movie_id": "7cbc7c3b-3b42-4e3c-9908-7bbc661c6659"}
    # Отправка сообщения в топик
    producer.send('user-stats', value=message)
    # Завершаем отправку
    producer.flush()

print("Message sent to Kafka.")
