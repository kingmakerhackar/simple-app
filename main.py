from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx, os, json, re

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_KEY = os.environ.get("GROQ_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

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
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7}
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

@app.get("/")
async def root():
    html = open("index.html").read()
    return HTMLResponse(html)

@app.post("/chat")
async def chat(req: ChatReq):
    messages = [{
        "role": "system",
        "content": f"""நீ BookChat AI. புத்தகம் பத்தி பேசும் ஒரு அறிவாளி நண்பன்.
கேட்கும் மொழியில் பதில் சொல் — தமிழில் கேட்டால் தமிழில், English-ல் கேட்டால் English-ல்.
புத்தகத்தில் இல்லாத விஷயங்களையும் உன் அறிவிலிருந்து சேர்த்து விரிவாக பதில் சொல்.
புத்தகத்தில் உள்ள characters, themes, lessons பத்தி ஆழமாக பேசு.

BOOK CONTENT:
{req.source_text[:5000]}"""
    }]
    for h in req.history[-10:]:
        messages.append({"role": h.get("role","user"), "content": h.get("content","")})
    messages.append({"role": "user", "content": req.question})
    answer = await call_groq(messages, max_tokens=800)
    return {"answer": answer}

@app.post("/generate-script")
async def generate_script(req: ScriptReq):
    
    lang_instructions = {
        "tamil": "முழுக்க முழுக்க தமிழில் மட்டும் எழுது. ஒரு வார்த்தையும் ஆங்கிலத்தில் வேண்டாம்.",
        "english": "Write entirely in English only.",
        "both": "தமிழும் English-உம் கலந்து எழுது — ஒரே வாக்கியத்தில் இரண்டும் கலைக்கலாம் (Tanglish style)."
    }
    
    length_instructions = {
        "short": """சுருக்கமாக எழுது:
- கதையின் மிக முக்கியமான 2-3 விஷயங்களை மட்டும் எடு
- அதை சுருக்கி சுருக்கி சொல்
- 150-200 words மட்டும்""",
        "medium": """சமநிலையாக எழுது:
- கதையை balance-ஆக cover பண்ணு
- சில creative additions சேர்க்கலாம்  
- 300-400 words""",
        "long": """விரிவாக கற்பனையோடு எழுது:
- கதையை முழுமையாக expand பண்ணு
- Characters-ஓட உணர்வுகளை ஆழமாக describe பண்ணு
- புத்தகத்தில் இல்லாத creative scenes சேர்க்கலாம்
- 600-800 words"""
    }

    if req.style == "solo":
        fmt = '{"type":"solo","title":"தலைப்பு","paragraphs":["பத்தி 1","பத்தி 2","பத்தி 3"]}'
        style_inst = "ஒரு narrator சொல்வது போல் எழுது"
    elif req.style == "duo":
        fmt = '{"type":"duo","title":"தலைப்பு","lines":[{"speaker":"அர்ஜுன்","text":"வசனம்"},{"speaker":"பிரியா","text":"வசனம்"}]}'
        style_inst = "அர்ஜுன் மற்றும் பிரியா இருவரும் கதை பத்தி பேசுவது போல் எழுது. குறைந்தது 8 lines."
    else:
        fmt = '{"type":"drama","title":"தலைப்பு","scene":"காட்சி விளக்கம்","lines":[{"character":"பெயர்","dialogue":"வசனம்","isStage":false}]}'
        style_inst = "கதையில் உள்ள characters-ஐ வைத்து நாடகம் எழுது. குறைந்தது 10 lines."

    prompt = f"""இந்தக் கதையை audio script-ஆக மாத்து.

மொழி: {lang_instructions.get(req.lang, lang_instructions['tamil'])}

நீளம்: {length_instructions.get(req.length, length_instructions['short'])}

Style: {style_inst}

கதை:
{req.source_text[:4000]}

IMPORTANT: ONLY return this exact JSON format, nothing else before or after:
{fmt}"""

    messages = [
        {"role": "system", "content": "You are a script writer. You must return ONLY a JSON object. No explanation, no markdown, no ```json``` blocks. Start your response with { and end with }"},
        {"role": "user", "content": prompt}
    ]
    
    raw = await call_groq(messages, max_tokens=2000)
    
    # Clean response
    raw = raw.strip()
    # Remove markdown if present
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'^```\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()
    
    # Find JSON object
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1:
        raw = raw[start:end+1]
    
    script = json.loads(raw)
    return {"script": script}
