import os
from dotenv import load_dotenv


load_dotenv()


KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
TOPIC_USER_STATS = os.getenv('TOPIC_USER_STATS', 'user-stats')


DB_CONFIG = {
    "database": os.getenv("DB_STATS_NAME", "db_stats"),
    "user": os.getenv("DB_STATS_USER", "db_stats"),
    "password": os.getenv("DB_STATS_PASSWORD", "db_stats"),
    "host": os.getenv("DB_STATS_HOST"),
    "port": os.getenv("DB_STATS_PORT", "5432"),
}

BATCH_SIZE = int(os.getenv('BATCH_SIZE', 100))