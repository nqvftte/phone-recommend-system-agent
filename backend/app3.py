from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from agent3 import llm  # get_agent() sẽ trả về Smartphone_Recommendation_Agent
from agent3 import Smartphone_Recommendation_Agent as agent
app = FastAPI()

# Cho phép frontend (Streamlit) kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"  # để phân biệt lịch sử từng user

async def run_agent_with_translation(agent, message: str) -> str:
    # Gọi agent (English prompt & description)
    english_response = await agent.run(message)

    # Dịch sang tiếng Việt
    translation_prompt = f"Translate the following text into natural Vietnamese:\n\n{english_response}"
    vietnamese_answer = llm.complete(translation_prompt).text

    return vietnamese_answer

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # Giữ history theo user_id (nếu muốn)
    config = {"configurable": {"thread_id": req.user_id}}

    # Gọi agent async + dịch sang tiếng Việt
    final_text = await run_agent_with_translation(agent, req.message)

    return {"response": final_text}
