import asyncio
import json

from aiokafka import AIOKafkaConsumer
from clicks.dao import ClicksDAO
from consumer.consumer import consume_clicks




async def main():
    consume = AIOKafkaConsumer(
        'user-stats',  # Топик
        bootstrap_servers='kafka:9092',  # Адрес брокера Kafka
        group_id='test-group',  # Группа консумеров
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='earliest'  # Начинать с начала, если нет смещения
    )
    await consume_clicks(consume, ClicksDAO)

if __name__ == "__main__":
    asyncio.run(main())