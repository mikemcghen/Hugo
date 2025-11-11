# Whisper Speech-to-Text Service
# GPU-accelerated OpenAI Whisper for local voice input

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    openai-whisper \
    faster-whisper \
    torch \
    torchaudio \
    fastapi \
    uvicorn[standard]

WORKDIR /app

# Copy service code
COPY server.py /app/

# Expose API port
EXPOSE 8001

# Download model on build (optional, can be done at runtime)
# RUN python3 -c "import whisper; whisper.load_model('base')"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
