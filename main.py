import os
import uuid
import shutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from gradio_client import Client, handle_file

app = FastAPI(title="danitts API", version="1.0.0")

# Allow cross-origin requests (so external bots/apps can talk to your API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to hold generated audio files
AUDIO_DIR = "static/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Your reference cloned voice file
REFERENCE_VOICE = "voice_reference.wav"

class SpeechRequest(BaseModel):
    text: str
    language: str = "English"

@app.post("/api/v1/tts")
async def generate_speech(data: SpeechRequest, request: Request):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty.")

    if not os.path.exists(REFERENCE_VOICE):
        raise HTTPException(status_code=500, detail=f"Reference voice file '{REFERENCE_VOICE}' not found on server.")

    try:
        # Connect to Qwen3-TTS / Voice Cloning engine on Hugging Face
        client = Client("Qwen/Qwen3-TTS")

        # Call voice cloning endpoint
        result = client.predict(
            ref_audio=handle_file(REFERENCE_VOICE),
            ref_text="",  # Optional transcript of your cloned voice sample
            use_xvector=True,
            target_text=data.text,
            language=data.language,
            model_size="0.6B",
            api_name="/clone_and_generate"
        )

        # Handle file returned by Gradio
        temp_audio_path = result[0] if isinstance(result, (tuple, list)) else result
        
        # Unique filename generation
        unique_id = uuid.uuid4().hex[:10]
        filename = f"danitts_{unique_id}.wav"
        destination_path = os.path.join(AUDIO_DIR, filename)

        # Copy generated file to public static folder
        shutil.copy(temp_audio_path, destination_path)

        # Generate full public link
        base_url = str(request.base_url).rstrip("/")
        audio_url = f"{base_url}/static/audio/{filename}"

        return {
            "status": "success",
            "text": data.text,
            "audio_url": audio_url,
            "filename": filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>danitts API is running!</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    