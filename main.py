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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Server-side API key — users need not provide their own
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_KEY", "")

class ChatRequest(BaseModel):
    source_text: str
    question: str
    history: list = []

class ScriptRequest(BaseModel):
    source_text: str
    lang: str
    length: str
    style: str

class TTSRequest(BaseModel):
    text: str
    voice: str = "tamil_male"

VOICES = {
    "tamil_male":    "ta-IN-ValluvarNeural",
    "tamil_female":  "ta-IN-PallaviNeural",
    "english_male":  "en-IN-PrabhatNeural",
    "english_female":"en-IN-NeerjaNeural",
}

@app.get("/")
def root():
    return {"status": "BookChat Backend running! 🎉"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        if not ANTHROPIC_KEY:
            raise HTTPException(status_code=500, detail="Server API key not configured")
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-script")
async def generate_script(req: ScriptRequest):
    try:
        if not ANTHROPIC_KEY:
            raise HTTPException(status_code=500, detail="Server API key not configured")
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        lang_map = {"tamil": "Tamil", "english": "English", "both": "Tamil and English mixed"}
        len_map = {
            "short": "2-3 minutes, key points only",
            "medium": "4-6 minutes, balanced",
            "long": "8-10 minutes, detailed"
        }
        if req.style == "solo":
            format_inst = '''Return JSON:
{"type":"solo","title":"...","paragraphs":["para1","para2"]}'''
        elif req.style == "duo":
            format_inst = '''Return JSON:
{"type":"duo","title":"...","lines":[{"speaker":"Arjun","text":"..."},{"speaker":"Priya","text":"..."}]}'''
        else:
            format_inst = '''Return JSON:
{"type":"drama","title":"...","scene":"...","lines":[{"character":"NAME","dialogue":"...","isStage":false}]}'''

        prompt = f"""Convert this content into an audio {req.style} script.
Language: {lang_map.get(req.lang, 'Tamil')}
Length: {len_map.get(req.length, 'medium')}

CONTENT:
{req.source_text[:4000]}

{format_inst}

Return ONLY valid JSON, nothing else."""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        import json, re
        raw = response.content[0].text.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'^```\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = re.sub(r',\s*}', '}', raw)
        raw = re.sub(r',\s*]', ']', raw)
        script = json.loads(raw)
        return {"script": script}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    try:
        voice = VOICES.get(req.voice, "ta-IN-ValluvarNeural")
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
        
