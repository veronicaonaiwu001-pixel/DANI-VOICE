from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from gradio_client import Client, handle_file
from pydantic import BaseModel
import os

app = FastAPI(title="danitts Studio")

# 1. Serve index.html when opening the root URL ("/")
@app.get("/", response_class=FileResponse)
async def read_index():
    # Make sure index.html is in your repo directory
    return FileResponse("index.html")

# 2. Your TTS API Request Model
class TTSRequest(BaseModel):
    text: str

# 3. Your Voice Synthesis Route
@app.post("/api/v1/tts")
async def generate_speech(request: TTSRequest):
    try:
        # Point to a public zero-shot TTS model (like F5-TTS or XTTS)
        client = Client("abidlabs/E2-F5-TTS")
        
        ref_path = os.path.join(os.path.dirname(__file__), "voice_reference.wav")
        
        result = client.predict(
            ref_audio_input=handle_file(ref_path),
            ref_text_input="",
            gen_text_input=request.text,
            model_choice="F5-TTS",
            remove_silence=False,
            api_name="/basic_tts"
        )
        return {"audio_path": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis error: {e}")
