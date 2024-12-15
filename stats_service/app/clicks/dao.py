
# Конфигурация для обработки данных
from collections import deque
import json
import logging
from uuid import UUID

import asyncpg

from config.config import BATCH_SIZE, DB_CONFIG



# Буфер для хранения записей
clicks_buffer = deque()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()  # Вывод логов в stdout
    ]
)

logger = logging.getLogger(__name__)

class ClicksDAO():
    @staticmethod
    async def insert_batch():
        pool: asyncpg.Pool = await asyncpg.create_pool(
            **DB_CONFIG,
            min_size=1,
            max_size=10,
        )
        try:
            """
            Функция для записи данных из буфера в базу данных.
            """
            global clicks_buffer
            if not clicks_buffer:
                return

            batch = []
            while clicks_buffer and len(batch) < BATCH_SIZE:
                batch.append(clicks_buffer.popleft())

            # Асинхронная вставка данных в базу
            async with pool.acquire() as connection:
                query = """
                    INSERT INTO user_movie (user_id, movie_id)
                    VALUES ($1, $2)
                    ON CONFLICT (user_id, movie_id) DO NOTHING
                """
                await connection.executemany(query, batch)
            logger.info(f"Inserted batch of {len(batch)} records")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
        finally:
            await pool.close()

    @classmethod
    async def process_message(cls, user_id, movie_id):
        """
        Обрабатывает сообщение из Kafka и добавляет его в буфер.
        """
        global clicks_buffer
        try:
            # Декодирование JSON
            logger.info(user_id, movie_id)
            
            # Добавление в буфер
            clicks_buffer.append((user_id, movie_id))
            if len(clicks_buffer) >= BATCH_SIZE:
                await ClicksDAO.insert_batch()
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error processing message: {e}", exc_info=True)