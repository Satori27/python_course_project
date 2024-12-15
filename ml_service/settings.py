import os
from dotenv import load_dotenv

load_dotenv()

DB_STATS_CONFIG = {
    "dbname": os.getenv("DB_STATS_NAME"),
    "user": os.getenv("DB_STATS_USER"),
    "password": os.getenv("DB_STATS_PASSWORD"),
    "host": os.getenv("DB_STATS_HOST"),
    "port": os.getenv("DB_STATS_PORT"),
}

DB_RECOMMEND_CONFIG = {
    "dbname": os.getenv("DB_RECOMMEND_NAME"),
    "user": os.getenv("DB_RECOMMEND_USER"),
    "password": os.getenv("DB_RECOMMEND_PASSWORD"),
    "host": os.getenv("DB_RECOMMEND_HOST"),
    "port": os.getenv("DB_RECOMMEND_PORT"),
}

POOL_MIN_CONN = int(os.getenv("POOL_MIN_CONN", 1))
POOL_MAX_CONN = int(os.getenv("POOL_MAX_CONN", 10))