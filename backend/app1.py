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
# translator = ChatTogether(
#     model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
#     temperature=0.7,
#     max_tokens=512,  # tránh vượt quá giới hạn token
# )
# # Prompt dịch sang tiếng Việt
# translate_prompt = ChatPromptTemplate.from_template("""
# Dịch nội dung sau sang tiếng Việt tự nhiên, giữ nguyên ý nghĩa và thông tin, chỉ hiện thị thông tin được dịch:
# "{text}"
# """)
# def translate_to_vietnamese(text: str) -> str:
#     prompt = translate_prompt.format_messages(text=text)
#     translation = translator.invoke(prompt)
#     return translation.content

smartphone_prompt = """
## Overview
You are a helpful AI assistant specialized in recommending smartphones to users based on their needs (e.g., gaming, photography, long battery life, budget).  
Your task is to understand the user’s intent, find relevant products using the retriever tool, and provide a final concise recommendation.

## Tools
You can only use the following tool:
- **product_retriever**: Search the smartphone database by keywords or product names.

## Rules
1. Always think step by step: first understand the user query, then decide if the tool is needed.
2. If the retriever is needed, follow the format:
   - `Thought:` Your reasoning  
   - `Action:` The tool name (`product_retriever`)  
   - `Action Input:` The exact search keyword or product name
3. Only respond with `Final Answer:` **after** receiving the Observation from the tool.
4. Do NOT make up product information. If the retriever returns nothing, say so.
5. Always reply in English unless the user explicitly requests another language.

## Examples

**Example 1 (retriever needed):**
User: *"Do you have a phone with strong battery for gaming?"*  
Thought: I should use the retriever to find phones with large batteries and good performance.  
Action: product_retriever  
Action Input: "long battery life gaming smartphone"

**Observation:** [list of matching products]  
Final Answer: Here are some smartphones with long-lasting batteries and good gaming performance: [summarized list].

---

**Example 2 (no retriever result):**  
User: *"Do you sell the XYZ Phone 12 Ultra?"*  
Thought: I need to check with the retriever for XYZ Phone 12 Ultra.  
Action: product_retriever  
Action Input: "XYZ Phone 12 Ultra"

**Observation:** [] (no results)  
Final Answer: Sorry, we currently don’t have XYZ Phone 12 Ultra in our catalog.

---

## Final Reminder
- Always follow the Thought → Action → Action Input → Observation → Final Answer structure.
- Never skip the tool call when product information is needed.
- Be concise but informative in the Final Answer.

User question: {input}
"""

def build_input_prompt(user_message, memory):
    history_data = memory.load_memory_variables({})["history"]

    # Nếu history là list, chuyển thành chuỗi
    if isinstance(history_data, list):
        history_text = "\n".join(
            f"{'User' if m.type=='human' else 'AI'}: {m.content}"
            for m in history_data
        )
    else:
        history_text = history_data

    if not history_text.strip():
        history_text = "No previous conversation."
    # Prompt chính
    smartphone_prompt = f"""
## Overview
You are a helpful AI assistant specialized in recommending smartphones to users based on their needs (e.g., gaming, photography, long battery life, budget).  
Your task is to understand the user’s intent, find relevant products using the retriever tool, and provide a final concise recommendation.

## Tools
You can only use the following tool:
- **product_retriever**: Search the smartphone database by keywords or product names.

## Rules
1. Always think step by step: first understand the user query, then decide if the tool is needed.
2. If the retriever is needed, follow the format:
   - `Thought:` Your reasoning  
   - `Action:` The tool name (`product_retriever`)  
   - `Action Input:` The exact search keyword or product name
3. Only respond with `Final Answer:` **after** receiving the Observation from the tool.
4. Do NOT make up product information. If the retriever returns nothing, say so.
5. Always reply in English unless the user explicitly requests another language.

## Examples

**Example 1 (retriever needed):**
User: *"Do you have a phone with strong battery for gaming?"*  
Thought: I should use the retriever to find phones with large batteries and good performance.  
Action: product_retriever  
Action Input: "long battery life gaming smartphone"

**Observation:** [list of matching products]  
Final Answer: Here are some smartphones with long-lasting batteries and good gaming performance: [summarized list].

---

**Example 2 (no retriever result):**  
User: *"Do you sell the XYZ Phone 12 Ultra?"*  
Thought: I need to check with the retriever for XYZ Phone 12 Ultra.  
Action: product_retriever  
Action Input: "XYZ Phone 12 Ultra"

**Observation:** [] (no results)  
Final Answer: Sorry, we currently don’t have XYZ Phone 12 Ultra in our catalog.

---

## Final Reminder
- Always follow the Thought → Action → Action Input → Observation → Final Answer structure.
- Never skip the tool call when product information is needed.
- Be concise but informative in the Final Answer.

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
    # final_result = translate_to_vietnamese(response["output"])
    # return {"response1": response["output"], "response": final_result}
    return {"response": response["output"]}

