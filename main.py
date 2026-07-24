import os
import uuid
import shutil
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from gradio_client import Client, handle_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("danitts")

app = FastAPI(
    title="danitts API",
    version="1.0.0",
    description="Text-to-Speech API powered by Qwen3-TTS Voice Cloning"
)

# Enable CORS for cross-origin requests from any client
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

# Helper function to auto-detect reference voice file
def get_reference_voice_file() -> str:
    possible_names = ["voice_reference.wav", "your_cloned_voice.wav", "voice.wav"]
    for name in possible_names:
        if os.path.exists(name):
            return name
    
    # Fallback search for any .wav file in the current directory
    for file in os.listdir("."):
        if file.endswith(".wav") and not file.startswith("speech_") and not file.startswith("danitts_"):
            return file
            
    return ""

class SpeechRequest(BaseModel):
    text: str
    language: str = "English"

@app.post("/api/v1/tts")
async def generate_speech(data: SpeechRequest, request: Request):
    text_content = data.text.strip()
    if not text_content:
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty.")

    voice_file = get_reference_voice_file()
    if not voice_file:
        raise HTTPException(
            status_code=500, 
            detail="No reference voice WAV file found on the server root directory."
        )

    logger.info(f"Synthesizing text: '{text_content[:30]}...' using voice file: {voice_file}")

    try:
        # Connect to Gradio Hugging Face Space
        client = Client("Qwen/Qwen3-TTS")

        # Execute prediction call
        # Passes positional args corresponding to the space interface
        try:
            result = client.predict(
                handle_file(voice_file), # Reference audio file
                "",                      # Reference transcript (optional)
                True,                    # Use x-vector for cloning
                text_content,            # Target text to pronounce
                data.language,           # Target language
                "0.6B",                  # Model size
                fn_index=1               # Voice cloning endpoint tab index
            )
        except Exception as api_err:
            logger.warning(f"fn_index call failed ({api_err}), attempting direct positional fallback...")
            result = client.predict(
                handle_file(voice_file),
                "",
                True,
                text_content,
                data.language,
                "0.6B"
            )

        # Process generated output path
        temp_audio_path = result[0] if isinstance(result, (tuple, list)) else result
        
        if not temp_audio_path or not os.path.exists(temp_audio_path):
            raise Exception("Gradio client returned an invalid audio file path.")

        # Generate unique filename for output
        unique_id = uuid.uuid4().hex[:10]
        filename = f"danitts_{unique_id}.wav"
        destination_path = os.path.join(AUDIO_DIR, filename)

        # Copy audio file to public static folder
        shutil.copy(temp_audio_path, destination_path)

        # Construct public URL based on incoming request host
        base_url = str(request.base_url).rstrip("/")
        audio_url = f"{base_url}/static/audio/{filename}"

        logger.info(f"Generated speech successfully: {audio_url}")

        return {
            "status": "success",
            "text": text_content,
            "audio_url": audio_url,
            "filename": filename
        }

    except Exception as e:
        logger.error(f"Speech synthesis exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis error: {str(e)}")

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
