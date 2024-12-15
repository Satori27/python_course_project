import os
from dotenv import load_dotenv


load_dotenv()

DB_CONFIG = {
    "database": os.getenv("DB_RECOMMEND_NAME"),
    "user": os.getenv("DB_RECOMMEND_USER"),
    "password": os.getenv("DB_RECOMMEND_PASSWORD"),
    "host": os.getenv("DB_RECOMMEND_HOST"),
    "port": os.getenv("DB_RECOMMEND_PORT"),
}

