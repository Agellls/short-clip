FROM python:3.11-slim

# Install FFmpeg, yt-dlp system dependencies
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    libass9 \
    libavcodec-extra \
    fonts-liberation \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY clip_server.py .
COPY .env* ./

# Create necessary directories
RUN mkdir -p assets temp

# Assets folder will be mounted as volume or created at runtime

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Start uvicorn server

CMD ["uvicorn", "clip_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]