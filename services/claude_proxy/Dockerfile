# Claude API Proxy Service
# Local proxy for Claude API with caching and rate limiting

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    anthropic \
    httpx \
    pydantic \
    redis

# Copy service code
COPY server.py /app/

EXPOSE 8003

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003"]
