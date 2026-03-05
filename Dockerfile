# Dockerfile
# Base image for KFP pipeline components and local development
#
# Build:
#   docker build -t kfp-pipeline-runner:latest .
#
# Run locally:
#   docker run --rm -v $(pwd)/src:/app/src kfp-pipeline-runner:latest python -m src.pipelines.training_pipeline

FROM python:3.11-slim

LABEL maintainer="ml-pipeline-kubeflow"
LABEL description="Kubeflow Pipelines component runner"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Default command: compile and submit the training pipeline
CMD ["python", "-m", "src.pipelines.training_pipeline"]
