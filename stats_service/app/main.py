import asyncio
import json

from aiokafka import AIOKafkaConsumer
from clicks.dao import ClicksDAO
from consumer.consumer import consume_clicks

import asyncpg
from config.config import DB_CONFIG


async def create_connection_pool() -> asyncpg.Pool:
    connection_pool: asyncpg.Pool = await asyncpg.create_pool(
        **DB_CONFIG,
        min_size=1,
        max_size=10,
    )
    print("pool created")
    return connection_pool


async def close_connection_pool( connection_pool: asyncpg.Pool):
    # Закрытие пула соединений
    await connection_pool.close()
    print("Connection pool closed.")

async def main():

    # SQL-запрос для вставки
    insert_query_users = """
    INSERT INTO users (name) VALUES ($1)
    """

    insert_query_movie = """
    INSERT INTO user_movie (user_id, movie_id) VALUES ($1, $2)
    """

    connection_pool: asyncpg.Pool = await create_connection_pool()
    try:
        for i in range(100):
            async with connection_pool.acquire() as connection:

                await connection.execute(insert_query_users, f"name{i}")
                await connection.execute(insert_query_movie, i, i)
                print(insert_query_movie, i, i)
    except Exception as e:
        print(f"Error: RecommendationDAO.GetRecommendation\n {e}")
    finally:
        await close_connection_pool(connection_pool)

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