# agent.py
import os
from langchain.tools import Tool, tool
from langchain.tools.retriever import create_retriever_tool
from langchain_together import ChatTogether
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer
from langchain.agents import initialize_agent, Tool
from langchain.agents import create_openai_functions_agent, AgentExecutor, AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor, LLMSingleActionAgent
from langchain.agents.mrkl.output_parser import MRKLOutputParser
from retriever import retriever  # retriever đã load FAISS hoặc vectorstore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import StrOutputParser

from dotenv import load_dotenv

# Nạp biến môi trường
load_dotenv()

# Đảm bảo API key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Khởi tạo LLM (Together)
llm = ChatTogether(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.7,
    max_tokens=512,  # tránh vượt quá giới hạn token
)


# --- Tạo tokenizer cho LLaMA ---
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

import json

def format_product(doc):
    meta = doc.metadata
    name = meta.get("Product Name", "Unknown Product")
    link = meta.get("Product URL", "#")
    current_price = meta.get("Current Price", "Unknown")
    old_price = meta.get("Old Price", "N/A")

    # Trường specs và summary
    basic_specs = meta.get("Basic Specs", "No specs")
    battery_summary = meta.get("Battery Summary", "No summary")
    connectivity_summary = meta.get("Connectivity Summary", "No summary")
    design_summary = meta.get("Design Summary", "No summary")
    camera_display_summary = meta.get("Camera & Display Summary", "No summary")

    # Xử lý trường màu sắc
    color_variants = meta.get("Color Variants", [])
    if isinstance(color_variants, str):
        try:
            color_variants = json.loads(color_variants.replace("'", '"'))
        except Exception as e:
            print("❌ Lỗi chuyển đổi Color Variants:", e)
            color_variants = []

    if isinstance(color_variants, list):
        colors = [c.get("color", "Unknown") for c in color_variants]
        color_str = ", ".join(colors) if colors else "N/A"
    else:
        color_str = "N/A"

    return (
        f"- [{name}]({link}) — Current Price: {current_price} | Old Price: {old_price}\n"
        f"  - Basic Specs: {basic_specs}\n"
        f"  - Camera & Display: {camera_display_summary}\n"
        f"  - Battery: {battery_summary}\n"
        f"  - Connectivity: {connectivity_summary}\n"
        f"  - Design: {design_summary}\n"
        f"  - Color Variants: {color_str}"
    )


def search_products_formatted(input_str: str) -> str:
    parts = [p.strip() for p in input_str.split("|")]
    query = parts[0] if parts else ""
    color_filter = parts[1].lower() if len(parts) > 1 else None

    docs = retriever.get_relevant_documents(query)
    results = []

    for doc in docs:
        # Lấy dữ liệu color variants
        raw_color = doc.metadata.get("Color Variants", "")
        if isinstance(raw_color, str):
            try:
                color_variants = json.loads(raw_color.replace("'", '"'))
            except Exception:
                color_variants = []
        elif isinstance(raw_color, list):
            color_variants = raw_color
        else:
            color_variants = []

        # Áp dụng filter nếu có
        if color_filter:
            color_list = [c.get("color", "").lower() for c in color_variants]
            if not any(color_filter in color for color in color_list):
                continue

        results.append(format_product(doc))

    return "\n\n".join(results) if results else "No matching products found."


retriever_tool = Tool(
    name="product_retriever",
    func=search_products_formatted,
    description=(
        "Use this tool to search for smartphones based on product name, price, specifications, or color.\n"
        "Input formats:\n"
        "- 'query' (e.g., a model name or keyword)\n"
        "- 'query|color' (e.g., 'Galaxy S24|blue') to filter by color\n\n"
        "Returns a list of matching products with details:\n"
        "- Product Name (linked)\n"
        "- Current Price and Old Price\n"
        "- Promotion info\n"
        "- Color Variants\n"
        "- Basic Specs\n"
        "- Camera & Display\n"
        "- Battery & Charging\n"
        "- Connectivity\n"
        "- Design & Material"
    )
)




#check_promotion
def check_promotion(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No matching products were found for your query."

    results = []

    for doc in docs:
        meta = doc.metadata
        name = meta.get("Product Name", "Unnamed product")
        promo_raw = meta.get("Promotion", "")
        promo = str(promo_raw).strip().lower()

        # Define what counts as "no promotion"
        invalid_promos = {"", "no", "no promotion", "none", "n/a", "-", "0"}
        if promo not in invalid_promos:
            results.append(f"- {name}: {promo_raw.strip()}")
        else:
            results.append(f"- {name}: No current promotions available.")

    return "\n".join(results)

promotion_tool = Tool(
    name="promotion_tool",
    func=check_promotion,
    description=(
        "Use this tool to check if a product has a current promotion.\n"
        "Input: the product name or a relevant search keyword.\n"
        "Output: A list of matching products with their promotion details, if available."
    )
)



# Tool: so sánh sản phẩm
def compare_products(query: str) -> str:
    """
    So sánh sản phẩm dựa trên tên trong query.
    Input ví dụ: "Samsung Galaxy A52, Xiaomi Redmi Note 10"
    """
    # Chuẩn hoá tên sản phẩm (tách theo dấu phẩy hoặc 'vs')
    names = [n.strip() for n in query.replace("vs", ",").split(",") if n.strip()]
    
    results = []
    for name in names:
        docs = retriever.get_relevant_documents(name)
        if docs:
            meta = docs[0].metadata
            price = meta.get("Giá hiện tại", "Không rõ")
            link = meta.get("Link", "#")
            results.append(f"**[{name}]({link})** — Giá: {price}\n{docs[0].page_content}")
        else:
            results.append(f"**{name}**: Không tìm thấy thông tin.")
    
    return "\n\n".join(results)

compare_products_tool = Tool(
    name="compare_products",
    func=compare_products,
    description=(
        "Dùng để so sánh nhiều sản phẩm (2 hoặc hơn) dựa trên tên. "
        "Input là chuỗi chứa tên sản phẩm, phân tách bởi dấu phẩy hoặc 'vs'."
    ),
)

def search_products_formatted(input_str: str) -> str:
    # Tách query và color nếu có
    parts = [p.strip() for p in input_str.split("|")]
    query = parts[0] if parts else ""
    color = parts[1] if len(parts) > 1 else None

    docs = retriever.get_relevant_documents(query)
    results = []

    for doc in docs:
        meta = doc.metadata
        # Lọc màu nếu có yêu cầu
        if color and meta.get("Màu sắc", "").strip().lower() != color.lower():
            continue
        results.append(format_product(doc))

    if not results:
        return "Xin lỗi, không tìm thấy sản phẩm phù hợp."
    return "\n\n".join(results)

product_search_tool = Tool(
    name="product_search_tool",
    func=search_products_formatted,
    description="Tìm điện thoại theo yêu cầu. Input: 'query|color' hoặc 'query'. Kết quả trả về danh sách sản phẩm."
)


# Danh sách tools
#tools = [retriever_tool, compare_products_tool, product_search_tool, promotion_tool]
tools = [retriever_tool, promotion_tool]

def get_num_tokens_from_messages(self, messages):
    text = "\n".join([m.content for m in messages])
    return len(tokenizer.encode(text))

# Patch method vào ChatTogether class (not instance)
ChatTogether.get_num_tokens_from_messages = get_num_tokens_from_messages

# --- Khởi tạo ConversationSummaryBufferMemory ---
memory = ConversationSummaryBufferMemory(
    llm=llm,  # bắt buộc truyền llm để nó tóm tắt khi vượt giới hạn
    max_token_limit=1024,  # giới hạn token cho lịch sử
    return_messages=True
)

#OPENAI_FUNCTIONS, ZERO_SHOT_REACT_DESCRIPTION
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True, 
)
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
# prompt = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template(system_prompt),
#     MessagesPlaceholder(variable_name="chat_history", optional=True),
#     HumanMessagePromptTemplate.from_template("{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
# ])
# agent.agent.llm_chain.prompt.template = smartphone_prompt
# llm_chain = LLMChain(llm=llm, prompt=prompt)
# agent.agent.llm_chain = llm_chain 
# print(agent.agent.llm_chain.prompt.template)

def get_agent():
    return agent




# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True,
#     memory=memory
# )

# prompt = PromptTemplate(
#     input_variables=["input", "agent_scratchpad"],
#     template="""
# Bạn là một AI Agent tư vấn điện thoại.
# Hãy suy nghĩ (reasoning) và lập luận bằng tiếng Việt.
# Luôn xuất reasoning theo dạng:
# "Suy nghĩ: ..."
# Sau đó trả lời người dùng với:
# "Trả lời: ..."

# Câu hỏi: {input}
# {agent_scratchpad}
# """
# )

# # 2. Kết hợp prompt + llm thành llm_chain
# llm_chain = LLMChain(llm=llm, prompt=prompt)

# # 3. Tạo ZeroShotAgent với llm_chain
# agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

# # 4. Tạo executor (có memory)
# agent_executor = AgentExecutor.from_agent_and_tools(
#     agent=agent,
#     tools=tools,
#     memory=memory,
#     verbose=True,
#     handle_parsing_errors=True
# )

# def run_agent(message: str):
#     """Hàm tiện ích gọi agent với input."""
#     response = agent_executor.invoke({"input": message})
#     if isinstance(response, dict) and "output" in response:
#         return response["output"]
#     if hasattr(response, "text"):
#         return response.text
#     return str(response)

