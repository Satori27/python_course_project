from uuid import UUID
from fastapi import APIRouter, HTTPException

from app.recommendation.dao import RecommendationDAO
from app.errors.internal import InternalError

router = APIRouter(prefix='/recommendation', tags=['Tender'])

@router.get("/{user_id}")
async def get_recommendation(user_id: UUID):
    result = await RecommendationDAO.GetRecommendation(user_id)

    if result==InternalError:
        raise HTTPException(status_code=500, detail=result)
    
    return result