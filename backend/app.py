# backend/app.py
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import create_agent, save_agent_memory

# Đọc API key từ biến môi trường
from dotenv import load_dotenv
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("⚠️ Missing TOGETHER_API_KEY in environment!")

app = FastAPI()

# Cho phép CORS (frontend có thể gọi API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dùng memory lâu dài (lưu JSON theo user_id)
PERSISTENT_MEMORY = True

# Input schema
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ResetRequest(BaseModel):
    user_id: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # Tạo agent cho user (có memory riêng)
    agent, memory, user_id = create_agent(req.user_id, persistent_memory=PERSISTENT_MEMORY)
    
    # Gửi query đến agent
    response = agent.run(req.message)

    # Lưu lại memory nếu persistent
    if PERSISTENT_MEMORY:
        save_agent_memory(user_id, memory)

    return {"reply": response}

@app.post("/reset")
async def reset_memory(req: ResetRequest):
    # Xoá file memory nếu tồn tại (reset hội thoại)
    from memory_store import reset_memory
    reset_memory(req.user_id)
    return {"status": "ok", "message": f"Memory for {req.user_id} reset."}

@app.get("/")
async def root():
    return {"status": "running", "model": "LLaMA3 (TogetherAI)"}
