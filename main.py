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
    language: str = "English"


@app.post("/api/v1/tts")
async def generate_speech(data: SpeechRequest, request: Request):
    text_content = data.text.strip()
    if not text_content:
        raise HTTPException(
            status_code=400, detail="Text parameter cannot be empty."
        )

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

        # Log exact API endpoints to Render console for debugging
        try:
            api_info = client.view_api(return_format="dict")
            logger.info(f"Target Space API Schema: {api_info}")
        except Exception as api_err:
            logger.warning(f"Could not fetch view_api info: {api_err}")

        result = None

        # Strategy A: Pass named arguments targeting standard TTS signature
        try:
            result = client.predict(
                ref_audio=handle_file(voice_file),
                ref_text="",
                target_text=text_content,
                target_lang=selected_lang,
            )
        except Exception as e_named:
            logger.warning(f"Named kwargs strategy failed: {e_named}")

        # Strategy B: Fallback to single text + audio standard positional prediction
        if not result:
            try:
                result = client.predict(
                    handle_file(voice_file),
                    text_content,
                    selected_lang,
                )
            except Exception as e_pos:
                logger.warning(f"Positional 3-arg strategy failed: {e_pos}")

        # Strategy C: Raw predict using default fn_index without explicit api_name parameter string
        if not result:
            result = client.predict(
                handle_file(voice_file),
                "",
                selected_lang,
                text_content,
                selected_lang,
                "0.6B",
            )

        # Process output audio path
        temp_audio_path = None
        if isinstance(result, (tuple, list)):
            temp_audio_path = result[0]
        elif isinstance(result, dict) and "name" in result:
            temp_audio_path = result["name"]
        else:
            temp_audio_path = str(result)

        if not temp_audio_path or not os.path.exists(temp_audio_path):
            raise Exception(f"Gradio client returned an invalid audio path: {temp_audio_path}")

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


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return """
    <html>
        <body style="background:#0d1117;color:#fff;font-family:sans-serif;padding:40px;text-align:center;">
            <h1>🎙️ danitts API Engine</h1>
            <p>FastAPI backend is running successfully!</p>
            <p>POST to <code>/api/v1/tts</code> to synthesize speech.</p>
        </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
