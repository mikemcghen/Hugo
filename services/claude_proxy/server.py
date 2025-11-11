"""
Claude API Proxy Service
-------------------------
Local proxy for Claude API with caching, rate limiting, and monitoring.
"""

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import anthropic

app = FastAPI(title="Hugo Claude Proxy")

# Initialize Anthropic client
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


class Message(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model: str = "claude-sonnet-4-5-20250929"
    messages: List[Message]
    max_tokens: int = 4096
    temperature: float = 1.0
    system: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "api_configured": ANTHROPIC_API_KEY is not None
    }


@app.post("/v1/messages")
async def create_message(request: CompletionRequest):
    """
    Proxy request to Claude API.

    Args:
        request: Completion request

    Returns:
        Claude API response
    """
    if not client:
        raise HTTPException(
            status_code=503,
            detail="Anthropic API key not configured"
        )

    try:
        # Convert messages to Anthropic format
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        # Call Claude API
        response = client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=request.system,
            messages=messages
        )

        # Return response
        return {
            "id": response.id,
            "type": response.type,
            "role": response.role,
            "content": [
                {"type": block.type, "text": block.text}
                for block in response.content
            ],
            "model": response.model,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }

    except anthropic.APIError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/usage")
async def get_usage_stats():
    """
    Get API usage statistics.

    TODO: Implement usage tracking and caching
    """
    return {
        "total_requests": 0,
        "total_tokens": 0,
        "cache_hits": 0,
        "cache_misses": 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
