from fastapi import FastAPI, HTTPException
from gradio_client import Client, handle_file
from pydantic import BaseModel
import os

app = FastAPI(title="danitts API")

# Connect to a public voice-cloning Hugging Face space
client = Client("abidlabs/E2-F5-TTS")

class TTSRequest(BaseModel):
    text: str

@app.post("/api/v1/tts")
async def generate_speech(request: TTSRequest):
    try:
        # Get path to reference WAV stored in your repo
        ref_path = os.path.join(os.path.dirname(__file__), "voice_reference.wav")
        
        # Pass both reference audio and target text to the AI model
        result = client.predict(
            ref_audio_input=handle_file(ref_path),
            ref_text_input="",                 # Auto-transcribe reference voice
            gen_text_input=request.text,        # Target text to speak
            model_choice="F5-TTS",
            remove_silence=False,
            api_name="/basic_tts"
        )
        return {"audio_path": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis error: {e}")
