FROM python:3.10.11-slim

RUN apt-get update && apt-get install -y gcc && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY deployment ./
COPY pipelines ./
COPY config.py ./config.py

EXPOSE 8080

CMD ["uvicorn", "deployment.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

#docker compose up --build
