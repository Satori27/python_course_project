import json
import logging
from aiokafka import AIOKafkaConsumer



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()  # Вывод логов в stdout
    ]
)

logger = logging.getLogger(__name__)


# Инициализация Kafka Consumer


async def consume_clicks(consume, dao):
    logger.info("Запуск консьюмера...")

    await consume.start()
    try:
        async for message in consume:
            logger.info(f"Получено сообщение: {message.value}")
            user_id: int = int(message.value['user_id'])
            movie_id: int = int(message.value['movie_id'])
            await dao.process_message(user_id, movie_id)
    except KeyboardInterrupt:
        logger.info("Консьюмер остановлен пользователем.")
    except Exception as e:
        logger.error(f"Консьюмер столкнулся с ошибкой: {e}", exc_info=True)
    finally:
        await dao.insert_batch()
        await consume.stop()
