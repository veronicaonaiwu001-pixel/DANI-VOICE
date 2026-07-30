import logging
import os
import shutil
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from gradio_client import Client, handle_file
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("danitts")

app = FastAPI(
    title="danitts API",
    version="1.0.0",
    description="Text-to-Speech API powered by Qwen3-TTS Voice Cloning",
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory structure setup
AUDIO_DIR = "static/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


def get_reference_voice_file() -> str:
    """Auto-detects reference WAV voice file in root directory."""
    possible_names = [
        "voice_reference.wav",
        "your_cloned_voice.wav",
        "voice.wav",
    ]
    for name in possible_names:
        if os.path.exists(name):
            return name

    for file in os.listdir("."):
        if (
            file.endswith(".wav")
            and not file.startswith("speech_")
            and not file.startswith("danitts_")
        ):
            return file

    return ""


class SpeechRequest(BaseModel):
    text: str
    language: str = "English"  # Default fallback


@app.post("/api/v1/tts")
async def generate_speech(data: SpeechRequest, request: Request):
    text_content = data.text.strip()
    if not text_content:
        raise HTTPException(
            status_code=400, detail="Text parameter cannot be empty."
        )

    # Sanitize and validate against Qwen3-TTS strict choices
    allowed_langs = [
        "Auto",
        "Chinese",
        "English",
        "Japanese",
        "Korean",
        "French",
        "German",
        "Spanish",
        "Portuguese",
        "Russian",
    ]

    selected_lang = (
        data.language if data.language in allowed_langs else "English"
    )

    voice_file = get_reference_voice_file()
    if not voice_file:
        raise HTTPException(
            status_code=500,
            detail="No reference voice WAV file found on the server root directory.",
        )

    logger.info(
        f"Synthesizing: '{text_content[:30]}...' | Lang: {selected_lang} | File: {voice_file}"
    )

    try:
        client = Client("Qwen/Qwen3-TTS")

        # Explicit prediction tuple for Gradio Qwen3-TTS API
        predict_args = (
            handle_file(voice_file),  # Reference audio
            "",  # Reference text prompt (optional)
            True,  # Use x-vector speaker embedding
            text_content,  # Target synthesis text
            selected_lang,  # Target language
            "0.6B",  # Model variant size
        )

        result = None
        endpoints_to_try = ["/voice_clone", "/predict", "/generate"]

        # Attempt call via target API name
        for endpoint in endpoints_to_try:
            try:
                result = client.predict(*predict_args, api_name=endpoint)
                if result:
                    break
            except Exception as ep_err:
                logger.debug(f"Endpoint '{endpoint}' failed: {ep_err}")
                continue

        # Positional index fallback if API names fail
        if not result:
            result = client.predict(*predict_args, fn_index=0)

        # Unpack returned file path string or tuple
        temp_audio_path = (
            result[0] if isinstance(result, (tuple, list)) else result
        )

        if not temp_audio_path or not os.path.exists(temp_audio_path):
            raise Exception("Gradio client returned an invalid audio file path.")

        # Persist generated asset to public static folder
        unique_id = uuid.uuid4().hex[:10]
        filename = f"danitts_{unique_id}.wav"
        destination_path = os.path.join(AUDIO_DIR, filename)

        shutil.copy(temp_audio_path, destination_path)

        base_url = str(request.base_url).rstrip("/")
        audio_url = f"{base_url}/static/audio/{filename}"

        return {
            "status": "success",
            "text": text_content,
            "audio_url": audio_url,
            "filename": filename,
        }

    except Exception as e:
        logger.error(f"Speech synthesis error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Speech synthesis error: {str(e)}"
        )
