from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx, os, json, re

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_KEY = os.environ.get("GROQ_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"

class ChatReq(BaseModel):
    source_text: str
    question: str
    history: list = []

class ScriptReq(BaseModel):
    source_text: str
    lang: str = "tamil"
    length: str = "short"
    style: str = "solo"

async def call_groq(messages, max_tokens=1000):
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": messages, "max_tokens": max_tokens}
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

@app.get("/")
async def root():
    html = open("index.html").read()
    return HTMLResponse(html)

@app.post("/chat")
async def chat(req: ChatReq):
    messages = [{"role": "system", "content": f"Answer questions about this content. Reply in Tamil if asked in Tamil, English if asked in English.\n\nCONTENT:\n{req.source_text[:5000]}"}]
    for h in req.history[-10:]:
        messages.append({"role": h.get("role","user"), "content": h.get("content","")})
    messages.append({"role": "user", "content": req.question})
    answer = await call_groq(messages)
    return {"answer": answer}

@app.post("/generate-script")
async def generate_script(req: ScriptReq):
    lang_map = {"tamil": "Tamil", "english": "English", "both": "Tamil and English mixed"}
    len_map = {"short": "2-3 minutes", "medium": "4-6 minutes", "long": "8-10 minutes"}
    if req.style == "solo":
        fmt = '{"type":"solo","title":"...","paragraphs":["..."]}'
    elif req.style == "duo":
        fmt = '{"type":"duo","title":"...","lines":[{"speaker":"Arjun","text":"..."},{"speaker":"Priya","text":"..."}]}'
    else:
        fmt = '{"type":"drama","title":"...","scene":"...","lines":[{"character":"NAME","dialogue":"...","isStage":false}]}'
    prompt = f"Convert to {req.style} audio script. Language: {lang_map.get(req.lang,'Tamil')}. Length: {len_map.get(req.length,'2-3 minutes')}.\nCONTENT: {req.source_text[:4000]}\nReturn ONLY this JSON: {fmt}"
    messages = [
        {"role": "system", "content": "You are a script writer. Return ONLY valid JSON, no extra text."},
        {"role": "user", "content": prompt}
    ]
    raw = await call_groq(messages, max_tokens=2000)
    raw = re.sub(r'^```json', '', raw.strip())
    raw = re.sub(r'^```', '', raw).replace('```', '').strip()
    script = json.loads(raw)
    return {"script": script}
