FROM python:3.10-slim-buster

# Устанавливаем системные зависимости
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install python-dotenv --timeout 1000 --no-cache-dir
RUN pip3 install numpy --timeout 1000 --no-cache-dir
RUN pip3 install pandas --timeout 1000 --no-cache-dir
RUN pip3 install psycopg2-binary==2.9.10 --timeout 1000 --no-cache-dir
RUN pip3 install tqdm --timeout 1000 --no-cache-dir
RUN pip3 install torch --timeout 1000 --no-cache-dir
RUN pip3 install pytorch-lightning --timeout 1000 --no-cache-dir
RUN pip3 install torchmetrics --timeout 1000 --no-cache-dir
RUN pip3 install pytorch-accelerated --timeout 1000 --no-cache-dir

COPY . .

CMD ["python3", "ml_part.py"]