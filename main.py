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

    # Force a valid allowed language string
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

        # Explicit prediction arguments matching Qwen3-TTS schema
        predict_args = (
            handle_file(voice_file),  # ref_audio
            "",  # ref_text
            True,  # use_xvector
            text_content,  # target_text
            selected_lang,  # language (MUST be valid choice)
            "0.6B",  # model_size
        )

        result = None
        endpoints_to_try = ["/voice_clone", "/predict", "/generate"]

        for endpoint in endpoints_to_try:
            try:
                result = client.predict(*predict_args, api_name=endpoint)
                if result:
                    break
            except Exception:
                continue

        if not result:
            result = client.predict(*predict_args, fn_index=0)

        temp_audio_path = (
            result[0] if isinstance(result, (tuple, list)) else result
        )

        if not temp_audio_path or not os.path.exists(temp_audio_path):
            raise Exception("Gradio client returned an invalid audio file path.")

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
