import os
import re
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_together import ChatTogether
from langchain.prompts import ChatPromptTemplate
from agent5 import get_agent, memory
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



# def extract_product_names_and_images(final_answer: str, intermediate_steps: list):
#     # 1. Trích danh sách (tên, link) từ final_answer dạng markdown
#     product_entries = re.findall(r'- \[(.*?)\]\((https?://[^\)]+)\)', final_answer)

#     # 2. Lấy phần observation chứa chi tiết mô tả (chứa cả image link)
#     detailed_text = ""
#     for action, observation in intermediate_steps:
#         if hasattr(action, 'tool') and action.tool == "product_retriever":
#             detailed_text = observation
#             break

#     # 3. Với mỗi sản phẩm, tìm đoạn mô tả và image link
#     results = []
#     for name, link in product_entries:
#         pattern = rf'- \[{re.escape(name)}\]\({re.escape(link)}\)(.*?)(?=\n- \[|$)'
#         match = re.search(pattern, detailed_text, re.DOTALL)
#         if match:
#             product_block = match.group(0)
#             img_match = re.search(r'Image Link:\s*(https?://\S+)', product_block)
#             image_link = img_match.group(1) if img_match else None
#             results.append({"name": name, "image_link": image_link})
#         else:
#             results.append({"name": name, "image_link": None})

#     return results

# def extract_product_names_and_images(final_answer: str, intermediate_steps: list):
#     import re

#     # 1. Tìm tất cả các sản phẩm từ final_answer
#     product_entries = re.findall(r'- \[(.*?)\]\((https?://[^\)]+)\)', final_answer)

#     # 2. Lấy phần observation chứa chi tiết mô tả (chứa cả image link)
#     detailed_text = ""
#     for action, observation in intermediate_steps:
#         if hasattr(action, 'tool') and action.tool == "product_retriever":
#             detailed_text = observation
#             break

#     # 3. Tìm tất cả các block sản phẩm trong detailed_text
#     product_blocks = re.findall(
#         r'- \[(.*?)\]\((https?://[^\)]+)\)(.*?)(?=\n- \[|$)',
#         detailed_text,
#         flags=re.DOTALL
#     )

#     # 4. Ghép theo tên và link
#     results = []
#     for name, link in product_entries:
#         image_link = None
#         for b_name, b_link, block in product_blocks:
#             if name == b_name and link == b_link:
#                 img_match = re.search(r'Image Link:\s*(https?://\S+)', block)
#                 if img_match:
#                     image_link = img_match.group(1)
#                 break
#         results.append({"name": name, "image_link": image_link})

#     return results


# def extract_product_names_and_images(final_answer: str, intermediate_steps: list):
#     results = []

#     # 1. Tìm observation từ product_retriever
#     detailed_text = ""
#     for action, observation in intermediate_steps:
#         if hasattr(action, "tool") and action.tool == "product_retriever":
#             detailed_text = observation
#             break

#     # 2. Tìm danh sách tên sản phẩm từ final_answer
#     markdown_entries = re.findall(r'\*\s+\[([^\]]+)\]\((https?://[^\)]+)\)', final_answer)
#     if markdown_entries:
#         product_names = [name for name, _ in markdown_entries]
#     else:
#         # Nếu không có markdown, fallback: tìm dòng bắt đầu bằng dấu * hoặc số thứ tự
#         fallback_entries = re.findall(r'(?:\*|\d+\.)\s+([^\n\-\(]+)', final_answer)
#         product_names = [name.strip() for name in fallback_entries if len(name.strip()) > 3]

#     # 3. Với mỗi sản phẩm, tìm image link tương ứng từ detailed_text
#     for name in product_names:
#         # Dùng regex không cần chính xác tuyệt đối, chỉ cần đoạn chứa tên
#         pattern = rf'{re.escape(name)}.*?(?=\n\S|\Z)'
#         match = re.search(pattern, detailed_text, re.DOTALL | re.IGNORECASE)

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
- **promotion_tool**: Only use when user mentions promotion to Check if a specific smartphone has any current promotions.

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
Final Answer: Here are some phones suitable for long movie sessions: [Product Name](https://product-link.com) - Price[...]

---

**Example 2 (check promotion):**  
User: *"Is there any promotion for the Galaxy S23?"*  
Thought: I should use the promotion_tool to check for deals on Galaxy S23.  
Action: promotion_tool  
Action Input: "Galaxy S23"

**Observation:** [promotion info or none]  
Final Answer: Samsung Galaxy S23 currently has the following offer: [...]

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

    # final_result = translate_to_vietnamese(response["output"])
    # return {"response1": response["output"], "response": final_result}
    return {"response": response["output"], "image_data":data}

