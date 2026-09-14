FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for OpenCV, Pydub, and PyAV (faster-whisper) compiled wheel steps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Optimize layer caching for Python requirements execution
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application layers into place
COPY ./app ./app

EXPOSE 8000 8501
