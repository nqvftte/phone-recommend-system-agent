import os
import re
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_together import ChatTogether
from langchain.prompts import ChatPromptTemplate
from agent6 import get_agent, memory
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



def extract_product_names_and_images(final_answer: str, intermediate_steps: list):
    results = []

    # 1. Lấy observation từ product_retriever
    detailed_text = ""
    for action, observation in intermediate_steps:
        if hasattr(action, "tool") and action.tool == "product_retriever":
            detailed_text = observation
            break

    # 2. Tìm danh sách tên sản phẩm từ final_answer
    markdown_entries = re.findall(r'\*\s+\[([^\]]+)\]\(https?://[^\)]+\)', final_answer)
    if markdown_entries:
        # Trường hợp có markdown: Lấy tên trong []
        product_names = [name.strip() for name in markdown_entries]

    else:
        # Trường hợp không có markdown
        # a) Thử bắt theo "* Tên sản phẩm" hoặc "1. Tên sản phẩm"
        fallback_entries = re.findall(r'(?:\*|\d+\.)\s+([^\n\-\(]+)', final_answer)
        product_names = [name.strip() for name in fallback_entries if len(name.strip()) > 3]

        # b) Nếu vẫn không bắt được tên nào => fallback: lấy từ observation
        if not product_names and detailed_text:
            observation_entries = re.findall(r'- \[([^\]]+)\]\(https?://[^\)]+\)', detailed_text)
            product_names = [name.strip() for name in observation_entries]

    # 3. Tìm link ảnh tương ứng từ observation (detailed_text)
    for name in product_names:
        # Tìm block chứa tên sản phẩm trong observation
        pattern = rf'{re.escape(name)}.*?(?=\n\S|\Z)'
        match = re.search(pattern, detailed_text, re.DOTALL | re.IGNORECASE)

        image_link = None
        if match:
            block = match.group(0)
            img_match = re.search(r'Image Link:\s*(https?://\S+)', block)
            if img_match:
                image_link = img_match.group(1)

        results.append({
            "name": name,
            "image_link": image_link
        })

    return results

# def extract_product_names_and_images(final_answer: str, intermediate_steps: list):
#     results = []

#     # 1. Lấy observation từ product_retriever
#     detailed_text = ""
#     for action, observation in intermediate_steps:
#         if hasattr(action, "tool") and action.tool == "product_retriever":
#             detailed_text = observation
#             break

#     # 2. Lấy tên sản phẩm từ final_answer (ưu tiên markdown dạng - [Tên](link))
#     product_names = re.findall(r'-\s*\[([^\]]+)\]\(https?://[^\)]+\)', final_answer)

#     # Nếu không có markdown, fallback sang dạng "* Tên" hoặc "1. Tên"
#     if not product_names:
#         fallback_entries = re.findall(r'(?:\*|\d+\.)\s+([^\n\-\(]+)', final_answer)
#         product_names = [name.strip() for name in fallback_entries if len(name.strip()) > 3]

#     # 3. Với mỗi sản phẩm từ final_answer, tìm link ảnh trong observation
#     for name in product_names:
#         pattern = rf'- \[{re.escape(name)}\]\([^\)]+\).*?(?=\n- |\Z)'
#         match = re.search(pattern, detailed_text, re.DOTALL)
        
#         image_link = None
#         if match:
#             block = match.group(0)
#             img_match = re.search(r'Image Link:\s*(https?://\S+)', block)
#             if img_match:
#                 image_link = img_match.group(1)

#         results.append({
#             "name": name,
#             "image_link": image_link
#         })

#     return results



# import re

# def extract_product_names_and_images(final_answer: str, intermediate_steps: list):
#     results = []

#     # 1. Lấy observation từ product_retriever
#     detailed_text = ""
#     for action, observation in intermediate_steps:
#         if hasattr(action, "tool") and action.tool == "product_retriever":
#             detailed_text = observation
#             break

#     # 2. Lấy danh sách sản phẩm từ observation (ưu tiên)
#     product_entries = re.findall(
#         r'- \[([^\]]+)\]\(https?://[^\)]+\).*?(?=\n- |\Z)', 
#         detailed_text, 
#         re.DOTALL
#     )

#     # Nếu không tìm thấy trong observation → fallback sang final_answer
#     if not product_entries:
#         product_entries = re.findall(
#             r'- \[([^\]]+)\]\(https?://[^\)]+\)', 
#             final_answer
#         )

#     # 3. Với mỗi sản phẩm, tìm link ảnh từ observation
#     for name in product_entries:
#         # Tìm block chứa sản phẩm
#         pattern = rf'- \[{re.escape(name)}\]\(https?://[^\)]+\)(.*?)(?=\n- |\Z)'
#         match = re.search(pattern, detailed_text, re.DOTALL)

#         image_link = None
#         if match:
#             block = match.group(0)
#             img_match = re.search(r'Image Link:\s*(https?://\S+)', block)
#             if img_match:
#                 image_link = img_match.group(1)

#         results.append({
#             "name": name,
#             "image_link": image_link
#         })

#     return results

translator = ChatTogether(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.7,
    max_tokens=512,  # tránh vượt quá giới hạn token
)

translate_prompt = ChatPromptTemplate.from_template("""
Translate the following content into natural Vietnamese, preserving the original meaning and information.
- Keep all links and Markdown formatting (e.g., [Product Name](URL)) unchanged.
- Only translate the descriptive text, not the links or URLs.
- Remove any mentions of internal tools (e.g., product_retriever, promotion_tool, compare_products) and rephrase naturally as if written by a human consultant.
- Display only the translated content without extra explanation or tool details.
- Only output the translated text itself. Do NOT add introductions like "Here is the translated content:" or quotation marks around the result.

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
- **compare_products**: Compare two or more smartphones by their names to see their specs and summaries.

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
Final Answer: Here are some phones suitable for long movie sessions: [Product Name](https://product-link.com)[...]

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
- In your Final Answer, if you have the product link, should format it like:  [Product Name](https://product-link.com)

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
    # response = agent.invoke({"input": full_prompt},return_intermediate_steps=True)
    print(response["intermediate_steps"])
    data = extract_product_names_and_images(response["output"], response["intermediate_steps"])
    print("data: ", data)
    for product in data:
        print(f"📱 {product['name']}\n🖼️ {product['image_link']}\n")

    final_result = translate_to_vietnamese(response["output"])
    # return {"response1": response["output"], "response": final_result}
    # return {"response": response["output"], "image_data":data}
    return {"response": final_result, "image_data":data}

class ResetRequest(BaseModel):
    user_id: str  # nếu bạn muốn phân biệt user, có thể dùng user_id

@app.post("/reset")
async def reset_memory(req: ResetRequest):
    memory.clear()  # hoặc memory.chat_memory.clear()
    return {"status": "success", "message": "Memory cleared"}