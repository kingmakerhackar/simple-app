from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import anthropic
import edge_tts
import asyncio
import os
import io

app = FastAPI()

# CORS — GitHub Pages frontend allow பண்றோம்
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MODELS =====

class ChatRequest(BaseModel):
    claude_key: str
    source_text: str
    question: str
    history: list = []

class ScriptRequest(BaseModel):
    claude_key: str
    source_text: str
    lang: str        # tamil / english / both
    length: str      # short / medium / long
    style: str       # solo / duo / drama

class TTSRequest(BaseModel):
    text: str
    voice: str = "ta-IN-ValluvarNeural"  # default Tamil male

# ===== VOICE MAP =====

VOICES = {
    "tamil_male":   "ta-IN-ValluvarNeural",
    "tamil_female": "ta-IN-PallaviNeural",
    "english_male": "en-IN-PrabhatNeural",
    "english_female": "en-IN-NeerjaNeural",
}

# ===== ROUTES =====

@app.get("/")
def root():
    return {"status": "BookChat Backend running! 🎉"}

@app.post("/chat")
async def chat(req: ChatRequest):
    """Claude-கிட்ட கேள்வி கேட்கும்"""
    try:
        client = anthropic.Anthropic(api_key=req.claude_key)

        system = f"""You are a helpful AI assistant. Answer questions about the provided content clearly.
Answer in Tamil if asked in Tamil, English if asked in English.

SOURCE CONTENT:
```
{req.source_text[:6000]}
```"""

        messages = req.history[-20:] + [{"role": "user", "content": req.question}]

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system,
            messages=messages
        )

        return {"answer": response.content[0].text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-script")
async def generate_script(req: ScriptRequest):
    """Script generate பண்ணும்"""
    try:
        client = anthropic.Anthropic(api_key=req.claude_key)

        lang_map = {"tamil": "Tamil", "english": "English", "both": "Tamil and English mixed"}
        len_map = {
            "short": "2-3 minutes, key points only",
            "medium": "4-6 minutes, balanced",
            "long": "8-10 minutes, detailed"
        }

        if req.style == "solo":
            format_inst = """Return JSON:
{
  "type": "solo",
  "title": "...",
  "paragraphs": ["paragraph1", "paragraph2", ...]
}"""
        elif req.style == "duo":
            format_inst = """Return JSON:
{
  "type": "duo",
  "title": "...",
  "lines": [
    {"speaker": "Arjun", "text": "..."},
    {"speaker": "Priya", "text": "..."}
  ]
}"""
        else:  # drama
            format_inst = """Return JSON:
{
  "type": "drama",
  "title": "...",
  "scene": "Scene description",
  "lines": [
    {"character": "NAME", "dialogue": "...", "isStage": false},
    {"character": "", "dialogue": "stage direction", "isStage": true}
  ]
}"""

        prompt = f"""Convert this content into an audio {req.style} script.
Language: {lang_map.get(req.lang, 'Tamil')}
Length: {len_map.get(req.length, 'medium')}

CONTENT:
{req.source_text[:4000]}

{format_inst}

Return ONLY valid JSON, no other text."""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        import json, re
        raw = response.content[0].text.strip()
        # Clean JSON
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'^```\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = re.sub(r',\s*}', '}', raw)
        raw = re.sub(r',\s*]', ']', raw)

        script = json.loads(raw)
        return {"script": script}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Edge TTS — text to audio"""
    try:
        voice = VOICES.get(req.voice, req.voice)
        communicate = edge_tts.Communicate(req.text, voice)

        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_buffer.seek(0)
        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=audio.mp3"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voices")
async def list_voices():
    """Available voices list"""
    return {"voices": VOICES}
