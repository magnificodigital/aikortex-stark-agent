FROM python:3.11-slim

# System deps pra Deepgram/livekit-agents (audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3
RUN poetry config virtualenvs.create false

# Cache dependencies layer separately
COPY pyproject.toml ./
RUN poetry install --no-interaction --no-ansi --no-root --without dev

# App code
COPY src ./src

# LiveKit worker connects out to LiveKit Cloud — no inbound port needed.
# Railway healthcheck via /health (mock — livekit-agents handles it internally)
CMD ["python", "-m", "src.main", "start"]
