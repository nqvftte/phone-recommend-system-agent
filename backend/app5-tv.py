import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_together import ChatTogether
from langchain.prompts import ChatPromptTemplate
from agent4 import get_agent, memory
from dotenv import load_dotenv

# Nạp biến môi trường
load_dotenv()

# Đảm bảo API key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

app = FastAPI()
agent = get_agent()

# Cho phép frontend (Streamlit) kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
translator = ChatTogether(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.7,
    max_tokens=512,  # tránh vượt quá giới hạn token
)
# Prompt dịch sang tiếng Việt
# translate_prompt = ChatPromptTemplate.from_template("""
# Dịch nội dung sau sang tiếng Việt tự nhiên, giữ nguyên ý nghĩa và thông tin, chỉ hiện thị thông tin được dịch:
# "{text}"
# """)
translate_prompt = ChatPromptTemplate.from_template("""
Translate the following content into natural Vietnamese, preserving the original meaning and information. Only display the translated content:
"{text}"
""")

def translate_to_vietnamese(text: str) -> str:
    prompt = translate_prompt.format_messages(text=text)
    translation = translator.invoke(prompt)
    return translation.content

def build_input_prompt(user_message, memory):
    history_data = memory.load_memory_variables({})["history"]

    if isinstance(history_data, list):
        history_text = "\n".join(
            f"{'User' if m.type=='human' else 'AI'}: {m.content}"
            for m in history_data
        )
    else:
        history_text = history_data

    if not history_text.strip():
        history_text = "No previous conversation."

    smartphone_prompt = f"""
## Overview
You are a helpful AI assistant specialized in recommending smartphones and checking promotions based on user requests.  
You should understand the user's intent, use appropriate tools (retriever or promotion), and provide concise, relevant responses.

## Tools
You can only use the following tools:
- **product_retriever**: Search the smartphone database by keywords or product names.
- **promotion_tool**: Check if a specific smartphone has any current promotions.

## Rules
1. Always think step by step: first understand the user's intent, then decide which tool to use (or both).
2. If a tool is needed, follow this format:
   - `Thought:` Your reasoning  
   - `Action:` The tool name (`product_retriever` or `promotion_tool`)  
   - `Action Input:` The exact keyword or product name
3. Only respond with `Final Answer:` **after** receiving the Observation from the tool.
4. Do NOT make up product or promotion information. If a tool returns no result, clearly say so.
5. Always reply in English unless the user explicitly requests another language.

## Examples

**Example 1 (use retriever):**  
User: *"I need a phone with a strong battery for watching movies."*  
Thought: I should use the retriever to find phones with good battery life and large displays.  
Action: product_retriever  
Action Input: "smartphone with long battery and big screen"

**Observation:** [list of matching products]  
Final Answer: Here are some phones suitable for long movie sessions: [...]

---

**Example 2 (check promotion):**  
User: *"Is there any promotion for the Galaxy S23?"*  
Thought: I should use the promotion_tool to check for deals on Galaxy S23.  
Action: promotion_tool  
Action Input: "Galaxy S23"

**Observation:** [promotion info or none]  
Final Answer: Samsung Galaxy S23 currently has the following offer: [...]

---

**Example 3 (no result from tool):**  
User: *"Do you sell the XYZ Phone 12 Ultra?"*  
Thought: I need to check with the retriever.  
Action: product_retriever  
Action Input: "XYZ Phone 12 Ultra"

**Observation:** []  
Final Answer: Sorry, we currently don’t have XYZ Phone 12 Ultra in our catalog.

---

## Final Reminder
- Always follow the Thought → Action → Action Input → Observation → Final Answer structure.
- Choose the right tool for the job.
- Never fabricate product or promotion details.
- Be concise but informative.

Conversation so far:
{history_text}

User question: {user_message}
"""
    return smartphone_prompt



# Request schema
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # Map message -> "input" cho agent
    full_prompt = build_input_prompt(req.message, memory)
    #response = agent.invoke({"input": smartphone_prompt.format(input = req.message)})
    response = agent.invoke({"input": full_prompt})
    final_result = translate_to_vietnamese(response["output"])
    # return {"response1": response["output"], "response": final_result}
    # return {"response": response["output"]}
    return {"response": final_result}

