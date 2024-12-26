import csv
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from src.repositories.user_repository import get_db
from src.repositories.movie_repository import Movie
from src.minio.minio_service import MinioClientWrapper

from src.dependencies import get_minio_client
import logging
from sqlalchemy.orm import sessionmaker, relationship, Session




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

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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

i=0
def upload_movie_local(
        title: str,
        file_path: str,  # Локальный путь к файлу
        genres: str,
        description: str = "",
):
    global i
    i+=1

    s3_key = f"0{i}/{os.path.basename(file_path)}"

    # Проверяем, что файл существует
    try:
        db = SessionLocal()
        # Загружаем файл в хранилище S3
        minio_client.upload_movie(s3_key, file_path)

        # Создаем запись в базе данных
        movie = Movie(
            title=title,
            description=description,
            s3_key=s3_key,
            genres=genres
        )
        db.add(movie)
        db.commit()
        db.refresh(movie)

        logging.info(f"Movie {title} uploaded successfully.")
    except Exception as e:
        logging.error(f"Failed to upload movie: {e}")
        raise

def upload_movies():
    with open("movie_rating.csv", mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                upload_movie_local(row["title"],"./SampleVideo_360x240_1mb.mp4", row["genres"] )
            except Exception as e:
                logging.error(f"Error reading CSV file: {e}")

            
upload_movies()
