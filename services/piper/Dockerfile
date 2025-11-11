# Piper Text-to-Speech Service
# GPU-accelerated Piper TTS for local voice output

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libsndfile1 \
    espeak-ng \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    piper-tts \
    fastapi \
    uvicorn[standard] \
    pydantic

WORKDIR /app

# Copy service code
COPY server.py /app/

# Create directory for voice models
RUN mkdir -p /app/models

# Download default voice model (optional, can be done at runtime)
# RUN wget -O /app/models/en_US-lessac-medium.onnx \
#     https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx

EXPOSE 8002

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
