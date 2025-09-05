# Dockerfile pour MLops Churn Classifier
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MLFLOW_EXPERIMENT_NAME=churn-exp

WORKDIR /app

COPY requirements.txt .
COPY src/ src/
COPY configs/ configs/
COPY data/ data/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Commande par défaut : lancer MLflow UI
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]
