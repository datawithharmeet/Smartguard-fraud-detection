# Base image
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install dependencies
RUN apt-get update -y \
    && apt-get install -y awscli \
    && pip install --no-cache-dir -r requirements.txt

# Entrypoint for different services via docker-compose
CMD ["python3", "app.py"]
