from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_PROMPT = """You are LexFin AI — a smart, professional assistant specializing in Finance and Law.

Your expertise covers:
- Personal finance: budgeting, savings, loans, EMIs, credit scores, investments, SIPs, mutual funds, tax planning (India-focused)
- Legal basics: consumer rights, contracts, tenant rights, employment law, FIR filing, RTI, basic civil/criminal distinctions

Guidelines:
- Always be helpful, clear, and concise
- Use simple language — avoid jargon unless asked
- Always add a disclaimer for legal/financial advice: "This is general information, not professional legal or financial advice. Please consult a certified expert for your specific situation."
- If a question is completely outside finance/law, politely redirect: "I specialize in Finance and Law topics. Could you ask me something in that domain?"
- Be conversational and warm, not robotic
"""

model = genai.GenerativeModel("gemini-2.0-flash")

sessions: dict[str, list] = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def root():
    return {"status": "LexFin AI is running 🚀"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = sessions.get(req.session_id, [])

    # Build conversation with system prompt injected as first user/model pair
    if not history:
        history = [
            {"role": "user", "parts": [SYSTEM_PROMPT + "\n\nAcknowledge you're ready."]},
            {"role": "model", "parts": ["Understood. I'm LexFin AI, ready to help with your Finance and Law questions!"]},
        ]

    chat_session = model.start_chat(history=history)
    response = chat_session.send_message(req.message)
    reply = response.text

    history.append({"role": "user", "parts": [req.message]})
    history.append({"role": "model", "parts": [reply]})
    sessions[req.session_id] = history

    return ChatResponse(reply=reply)

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared"}
