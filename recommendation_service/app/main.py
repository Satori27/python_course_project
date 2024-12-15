from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.recommendation.router import router as recommendation_router
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)


app.include_router(recommendation_router)