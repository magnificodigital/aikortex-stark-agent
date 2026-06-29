FROM python:3.11-slim

# System deps pra Deepgram/livekit-agents (audio processing + PyAV compile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CACHEBUST: incrementa pra forcar invalidacao de cache no Railway
# quando preciso re-rodar pip install sem reusar layer velha
ARG CACHEBUST=2

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3
RUN poetry config virtualenvs.create false

# Belt-and-suspenders: instala av explicitamente PRIMEIRO via pip antes do
# poetry. Se Poetry pular (cache, lock-file weirdness, etc), av ja esta.
RUN pip install --no-cache-dir av==12.3.0

# Cache dependencies layer separately
COPY pyproject.toml ./
RUN poetry install --no-interaction --no-ansi --no-root --without dev

# App code
COPY src ./src

# LiveKit worker connects OUT no LiveKit Cloud — sem porta inbound
CMD ["python", "-m", "src.main", "start"]
