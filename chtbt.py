import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
import datetime
import json

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("API key missing")

app = FastAPI(title="microgrid assistant bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=GROQ_API_KEY)


class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    role: str = "user"

class ChatResponse(BaseModel):
    conversation_id: str
    reply: str

class ItineraryRequest(BaseModel):
    conversation_id: str
    interests: List[str]
    days: int
    start_date: Optional[str] = None
    must_visit: Optional[List[str]] = None

class ItineraryDay(BaseModel):
    date: str
    activities: List[str]
    approx_cost: float

class ItineraryResponse(BaseModel):
    conversation_id: str
    itinerary: List[ItineraryDay]
    summary: str


# PLACE_DATABASE = {
#     "Netarhat": {"type":"nature","time_required":1,"approx_cost":2000},
#     "Patratu": {"type":"scenic","time_required":0.5,"approx_cost":500},
#     "Betla National Park": {"type":"wildlife","time_required":1.5,"approx_cost":3000},
#     "Hundru Falls": {"type":"waterfall","time_required":0.5,"approx_cost":300},
#     "Deoghar": {"type":"pilgrimage","time_required":1,"approx_cost":800}
# }


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

FAQ_DATASET = load_jsonl("dataset.jsonl")


class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful microgrid monitoring assistant, providing information about microgrid systems, their components, and best practices. Use the following FAQ data to assist users:\n" + json.dumps(FAQ_DATASET)}
        ]
        self.active: bool = True
        self.itinerary: Optional[List[ItineraryDay]] = None

conversations: Dict[str, Conversation] = {}

def get_or_create_conversation(conversation_id: str) -> Conversation:
    if conversation_id not in conversations:
        conversations[conversation_id] = Conversation()
    return conversations[conversation_id]


def query_groq_api(conversation: Conversation) -> str:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversation.messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        
        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with Groq API: {str(e)}")


def simple_schedule(interests: List[str], days: int, must_visit: Optional[List[str]]=None):
    picks = []
    if must_visit:
        for p in must_visit:
            if p in PLACE_DATABASE:
                picks.append((p, PLACE_DATABASE[p]))
    
    for pname, meta in PLACE_DATABASE.items():
        if len(picks) >= days*2: break
        if pname in (must_visit or []): continue
        for it in interests:
            if it.lower() in meta["type"] or it.lower() in pname.lower():
                picks.append((pname, meta))
                break
    
    for pname, meta in PLACE_DATABASE.items():
        if len(picks) >= days*2: break
        if not any(pname==x[0] for x in picks): picks.append((pname,meta))

    
    day_chunks = [picks[i:i+2] for i in range(0, min(len(picks), days*2), 2)]
    return day_chunks

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Chatbot API is running"}

@app.post("/chat/", response_model=ChatResponse)
async def chat(input: ChatRequest):
    conversation = get_or_create_conversation(input.conversation_id)
    if not conversation.active:
        raise HTTPException(status_code=400, detail="The chat session has ended.")
    
    conversation.messages.append({"role": input.role, "content": input.message})
    reply = query_groq_api(conversation)
    conversation.messages.append({"role": "assistant", "content": reply})
    
    return ChatResponse(conversation_id=input.conversation_id, reply=reply)

@app.post("/plan/", response_model=ItineraryResponse)
async def plan_itinerary(req: ItineraryRequest):
    conversation = get_or_create_conversation(req.conversation_id)
    start_date = datetime.date.fromisoformat(req.start_date) if req.start_date else datetime.date.today()
    
    day_chunks = simple_schedule(req.interests, req.days, req.must_visit)
    
    itinerary = []
    current = start_date
    for chunk in day_chunks:
        activities = []
        est_cost = 0
        for place, meta in chunk:
            activities.append(f"Visit {place} — time: {meta['time_required']} day(s).")
            est_cost += meta.get('approx_cost', 0)
        itinerary.append(ItineraryDay(date=current.isoformat(), activities=activities, approx_cost=est_cost))
        current += datetime.timedelta(days=1)
    
    conversation.itinerary = itinerary
    summary = f"{req.days}-day itinerary created with {len(day_chunks)} grouped stops."
    
    return ItineraryResponse(conversation_id=req.conversation_id, itinerary=itinerary, summary=summary)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
