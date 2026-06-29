FROM python:3.11-slim

# System deps pro PyAV (FFmpeg bindings) — wheel pre-buildado deve resolver
# mas mantemos libavcodec-dev como fallback se cair em compile-from-source.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Print marker no build pra confirmar visualmente que e build novo
RUN echo "=== Stark Agent build $(date) — v3 with av+pip ==="

# Install deps via pip direto — sem Poetry. Mais simples, sem lock files,
# sem layer cache confusion. requirements.txt com versoes pinadas.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Verifica que av instalou (se falhar, build quebra aqui em vez de runtime)
RUN python -c "import av; print(f'av OK: version={av.__version__}')"

# App code
COPY src ./src

# Verifica imports da livekit.agents (catch errors at build, not runtime)
RUN python -c "from livekit.agents import cli, WorkerOptions; print('livekit.agents OK')"

CMD ["python", "-m", "src.main", "start"]
