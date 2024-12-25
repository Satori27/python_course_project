import logging
from fastapi import APIRouter, HTTPException

from app.recommendation.dao import RecommendationDAO
from app.errors.internal import InternalError

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/recommendation', tags=['Tender'])

@router.get("/{user_id}")
async def get_recommendation(user_id: int):
    logger.info(f"recommendation_router.get_recommendation user_id:{user_id}")
    result = await RecommendationDAO.GetRecommendation(user_id)

    if result==InternalError:
        raise HTTPException(status_code=500, detail=result)
    logger.info(f"recommendation_router.get_recommendation result:{result}")

    return result