import csv
import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.repositories.movie_repository import Movie
from src.minio.minio_service import MinioClientWrapper

# Конфигурация окружения
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio123")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "movies")

DB_USER = os.getenv("POSTGRES_USER", "myuser")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mypassword")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "mydatabase")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Инициализация SQLAlchemy
engine = create_engine(DATABASE_URL, echo=False)  # echo=False для уменьшения логов
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Инициализация MinIO клиента
try:
    minio_client = MinioClientWrapper(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        bucket_name=MINIO_BUCKET
    )
    logging.info("MinIO клиент инициализирован.")
except Exception as e:
    logging.error(f"Не удалось инициализировать MinIO клиент: {e}")

def upload_movies(csv_file_path: str, file_path: str):
    """
    Массово загружает фильмы из CSV файла и добавляет их в базу данных.
    :param csv_file_path: Путь к CSV файлу.
    :param file_path: Путь к локальному видеофайлу.
    """
    # Генерируем S3 ключ и загружаем файл в S3
    s3_key = f"0/{os.path.basename(file_path)}"
    try:
        minio_client.upload_movie(s3_key, file_path)
    except Exception as e:
        logging.error(f"Failed to upload file to S3: {e}")
        return

    # Читаем данные из CSV и собираем записи для массовой вставки
    movies_data = []
    try:
        with open(csv_file_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                print("Data: ", row)
                movies_data.append({
                    "title": row["title"],
                    "description": "",
                    "s3_key": s3_key,
                    "genres": row["genres"]
                })
    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_file_path}")
        return
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return

    # Массовая вставка в базу данных
    if movies_data:
        with SessionLocal() as db:
            try:
                db.execute(
                    Movie.__table__.insert(),
                    movies_data
                )
                db.commit()
                logging.info(f"Successfully added {len(movies_data)} movies.")
            except Exception as e:
                db.rollback()
                logging.error(f"Failed to insert movies into the database: {e}")

# Запуск загрузки
upload_movies("movie_rating.csv", "./SampleVideo_360x240_1mb.mp4")
