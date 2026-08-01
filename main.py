import os
import logging
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ------------------------------------------------------------------------------
# Logging & Setup
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("danitts")

app = FastAPI(title="danitts Studio API")

# Live Colab Backend URL
COLAB_BACKEND_URL = "https://duchess-festivity-scalding.ngrok-free.dev"

# ------------------------------------------------------------------------------
# Request Schemas
# ------------------------------------------------------------------------------
class TTSRequest(BaseModel):
    text: str

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serves the frontend index.html UI at the root path."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found in repository.")

@app.post("/api/v1/tts")
async def generate_speech(request: TTSRequest):
    """Forwards text prompt & voice_reference.wav to Colab GPU backend."""
    try:
        ref_path = os.path.join(os.path.dirname(__file__), "voice_reference.wav")
        if not os.path.exists(ref_path):
            raise HTTPException(status_code=400, detail="voice_reference.wav missing from repo root.")

        logger.info(f"Forwarding prompt '{request.text}' to Colab GPU...")

        # Post voice_reference.wav and prompt text to Colab FastAPI server
        with open(ref_path, "rb") as f:
            files = {"ref_audio": ("voice_reference.wav", f, "audio/wav")}
            data = {"text": request.text}
            
            response = requests.post(
                f"{COLAB_BACKEND_URL}/synthesize",
                data=data,
                files=files,
                headers={"ngrok-skip-browser-warning": "true"}
            )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Colab GPU error: {response.text}")

        # Save returned WAV output locally on Render
        output_path = os.path.join(os.path.dirname(__file__), "output.wav")
        with open(output_path, "wb") as out_f:
            out_f.write(response.content)

        return {
            "audio_url": "/api/v1/audio",
            "audio_path": output_path
        }

    except Exception as e:
        logger.error(f"Synthesis proxy error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis error: {str(e)}")

@app.get("/api/v1/audio")
async def get_audio():
    """Streams generated WAV file back to frontend player."""
    output_path = os.path.join(os.path.dirname(__file__), "output.wav")
    if os.path.exists(output_path):
        return FileResponse(output_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio file not found.")
