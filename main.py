import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from gradio_client import Client, handle_file
from pydantic import BaseModel

# ------------------------------------------------------------------------------
# Logging & Setup
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("danitts")

app = FastAPI(title="danitts Studio API")

# Initialize Client with the inspected HF space
TTS_CLIENT = Client("mrfakename/E2-F5-TTS")

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
    """Serves the danitts Studio UI at the root path."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found in repository.")

@app.post("/api/v1/tts")
async def generate_speech(request: TTSRequest):
    """Synthesizes speech from prompt text using voice_reference.wav."""
    try:
        ref_path = os.path.join(os.path.dirname(__file__), "voice_reference.wav")
        if not os.path.exists(ref_path):
            raise HTTPException(status_code=400, detail="voice_reference.wav missing from repository.")

        logger.info(f"Generating speech for prompt: '{request.text}'")

        # Matched precisely to Client.predict() schema from Colab
        result = TTS_CLIENT.predict(
            ref_audio=handle_file(ref_path),
            ref_text="",
            gen_text=request.text,
            remove_silence=False,
            api_name="/predict"
        )

        logger.info("Speech synthesis completed successfully.")
        return {"audio_path": result}

    except Exception as e:
        logger.error(f"Speech synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis error: {str(e)}")
