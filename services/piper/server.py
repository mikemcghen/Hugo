"""
Piper Text-to-Speech Service
-----------------------------
FastAPI service exposing Piper TTS for voice synthesis.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import tempfile
import os
import subprocess

app = FastAPI(title="Hugo Piper Service")

# Voice configuration
DEFAULT_VOICE = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
MODELS_DIR = "/app/models"


class TTSRequest(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE
    speed: float = 1.0


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "default_voice": DEFAULT_VOICE}


@app.get("/voices")
async def list_voices():
    """List available voice models"""
    try:
        voices = [f for f in os.listdir(MODELS_DIR) if f.endswith(".onnx")]
        return {"voices": voices}
    except Exception as e:
        return {"voices": [], "error": str(e)}


@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text.

    Args:
        request: TTS request with text and voice settings

    Returns:
        WAV audio file
    """
    try:
        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            output_path = tmp.name

        # Construct piper command
        model_path = os.path.join(MODELS_DIR, f"{request.voice}.onnx")

        # TODO: Use piper-tts Python API if available, or subprocess
        # For now, using subprocess as placeholder

        # Example command: echo "text" | piper --model model.onnx --output_file out.wav
        cmd = [
            "piper",
            "--model", model_path,
            "--output_file", output_path
        ]

        # Run synthesis
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stdout, stderr = process.communicate(input=request.text.encode('utf-8'))

        if process.returncode != 0:
            raise Exception(f"Piper failed: {stderr.decode()}")

        # Return audio file
        def iterfile():
            with open(output_path, 'rb') as f:
                yield from f
            os.unlink(output_path)

        return StreamingResponse(iterfile(), media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
