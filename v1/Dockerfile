# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps needed by scipy/sklearn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# copy app code
COPY app /app/app
COPY src /app/src

# copy models folder (place your pickles into ./models before building)
COPY models /app/models

ENV MODEL_DIR=/app/models
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.deploy:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
