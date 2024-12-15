FROM python:3.10-slim-buster

# Устанавливаем системные зависимости
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD ["python3", "ml_part.py"]