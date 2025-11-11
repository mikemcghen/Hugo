"""
Whisper Speech-to-Text Service
-------------------------------
FastAPI service exposing Whisper for voice transcription.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os

app = FastAPI(title="Hugo Whisper Service")

# Load model on startup
# Options: tiny, base, small, medium, large
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
model = None


@app.on_event("startup")
async def load_model():
    """Load Whisper model on startup"""
    global model
    print(f"Loading Whisper model: {MODEL_SIZE}")
    model = whisper.load_model(MODEL_SIZE)
    print("Model loaded successfully")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": MODEL_SIZE}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = "en",
    task: str = "transcribe"
):
    """
    Transcribe audio file to text.

    Args:
        file: Audio file (wav, mp3, m4a, etc.)
        language: Language code (default: en)
        task: 'transcribe' or 'translate'

    Returns:
        JSON with transcription text and metadata
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Transcribe
        result = model.transcribe(
            tmp_path,
            language=language,
            task=task,
            fp16=False  # Set to True if GPU available
        )

        # Clean up
        os.unlink(tmp_path)

        return JSONResponse({
            "text": result["text"],
            "language": result.get("language"),
            "segments": result.get("segments", []),
            "model": MODEL_SIZE
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
