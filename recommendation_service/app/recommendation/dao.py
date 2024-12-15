from uuid import UUID
from app.database.conn import close_connection_pool, create_connection_pool
import asyncpg
from app.errors.internal import InternalError


class RecommendationDAO():
    async def GetRecommendation(user_id: UUID):
        connection_pool: asyncpg.Pool = await create_connection_pool()
        try:
            async with connection_pool.acquire() as connection:
                query = """
                SELECT movie_id FROM user_recommendations WHERE user_id = $1
                """

                rows = await connection.fetch(query, user_id)

                return rows
        except Exception as e:
            print(f"Error: RecommendationDAO.GetRecommendation\n {e}")
            return InternalError
        finally:
            await close_connection_pool(connection_pool)