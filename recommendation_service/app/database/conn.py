import asyncpg
from app.config.config import DB_CONFIG

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