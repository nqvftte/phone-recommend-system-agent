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
import re
import json


from dotenv import load_dotenv

# Nạp biến môi trường
load_dotenv()

# Đảm bảo API key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Khởi tạo LLM (Together) meta-llama/Llama-3-8b-chat-hf meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
llm = ChatTogether(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.7,
    max_tokens=1024,  # tránh vượt quá giới hạn token
)


# --- Tạo tokenizer cho LLaMA ---
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

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

    image_link = "N/A"
    if isinstance(color_variants, list):
        colors = [c.get("color", "Unknown") for c in color_variants]
        color_str = ", ".join(colors) if colors else "N/A"

        if color_variants:
            image_link = color_variants[0].get("image", "N/A")
    else:
        color_str = "N/A"

    return (
        f"- [{name}]({link}) — Current Price: {current_price} | Old Price: {old_price}\n"
        f"  - Basic Specs: {basic_specs}\n"
        f"  - Camera & Display: {camera_display_summary}\n"
        f"  - Battery: {battery_summary}\n"
        f"  - Connectivity: {connectivity_summary}\n"
        f"  - Design: {design_summary}\n"
        f"  - Color Variants: {color_str}\n"
        f"  - Image Link: {image_link}"
    )



# def search_products_formatted(input_str: str) -> str:
#     parts = [p.strip() for p in input_str.split("|")]
#     query = parts[0] if parts else ""
#     color_filter = parts[1].lower() if len(parts) > 1 else None

#     docs = retriever.get_relevant_documents(query)
#     results = []

#     for doc in docs:
#         # Lấy dữ liệu color variants
#         raw_color = doc.metadata.get("Color Variants", "")
#         if isinstance(raw_color, str):
#             try:
#                 color_variants = json.loads(raw_color.replace("'", '"'))
#             except Exception:
#                 color_variants = []
#         elif isinstance(raw_color, list):
#             color_variants = raw_color
#         else:
#             color_variants = []

#         # Áp dụng filter nếu có
#         if color_filter:
#             color_list = [c.get("color", "").lower() for c in color_variants]
#             if not any(color_filter in color for color in color_list):
#                 continue

#         results.append(format_product(doc))

#     return "\n\n".join(results) if results else "No matching products found."


def search_products_formatted(input_str: str) -> str:
    # 1. Tách màu (nếu có)
    parts = [p.strip() for p in input_str.split("|")]
    query = parts[0] if parts else ""
    color_filter = parts[1].lower() if len(parts) > 1 else None

    # 2. Parse RAM, Storage, Price từ query
    ram_match = re.search(r"(\d+)\s*GB\s*RAM", query, re.IGNORECASE)
    storage_match = re.search(r"(\d+)\s*GB\s*(storage|rom)?", query, re.IGNORECASE)
    price_match = re.search(r"under\s*([\d\.]+)\s*(million|trieu)", query, re.IGNORECASE)

    ram_required = int(ram_match.group(1)) if ram_match else None
    storage_required = int(storage_match.group(1)) if storage_match else None
    price_limit = float(price_match.group(1)) * 1_000_000 if price_match else None

    # 3. Semantic search trước
    docs = retriever.get_relevant_documents(query)
    results = []

    for doc in docs:
        meta = doc.metadata

        # 4. Lọc theo RAM/Storage/Price
        if ram_required and meta.get("RAM") and meta["RAM"] < ram_required:
            continue
        if storage_required and meta.get("Storage") and meta["Storage"] < storage_required:
            continue
        if price_limit and meta.get("Current Price Numeric") and meta["Current Price Numeric"] > price_limit:
            continue

        # 5. Lọc màu nếu có
        raw_color = meta.get("Color Variants", "")
        color_variants = json.loads(raw_color.replace("'", '"')) if isinstance(raw_color, str) else raw_color or []
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
        "Only Use this tool when user mention promotion to check if a product has a current promotion.\n"
        "Input: the product name or a relevant search keyword.\n"
        "Output: A list of matching products with their promotion details, if available."
        "When user mention promotion, return promotion to the final answer"
    )
)


# def compare_products(query: str) -> str:
#     # Chuẩn hóa tên sản phẩm
#     names = [n.strip() for n in query.replace("vs", ",").split(",") if n.strip()]
    
#     # Lấy metadata cho mỗi sản phẩm
#     products = []
#     for name in names:
#         docs = retriever.get_relevant_documents(name)
#         if docs:
#             meta = docs[0].metadata
#             products.append({
#                 "name": meta.get("Product Name", name),
#                 "price": meta.get("Current Price", "Unknown"),
#                 "battery": meta.get("Battery Summary", "N/A"),
#                 "display": meta.get("Camera & Display Summary", "N/A"),
#                 "design": meta.get("Design Summary", "N/A")
#             })
#         else:
#             products.append({"name": name, "price": "Not found", "battery": "-", "display": "-", "design": "-"})

#     # Nếu ít hơn 2 sản phẩm, báo lỗi
#     if len(products) < 2:
#         return "Please provide at least two products to compare."

#     # Tạo bảng Markdown
#     headers = "| Feature | " + " | ".join([p["name"] for p in products]) + " |"
#     separator = "|" + "---|" * (len(products) + 1)
#     rows = [
#         "| Price | " + " | ".join([p["price"] for p in products]) + " |",
#         "| Battery | " + " | ".join([p["battery"] for p in products]) + " |",
#         "| Display | " + " | ".join([p["display"] for p in products]) + " |",
#         "| Design | " + " | ".join([p["design"] for p in products]) + " |",
#     ]

#     return "\n".join([headers, separator] + rows)

def compare_products(query: str) -> str:
    # Chuẩn hóa tên sản phẩm
    names = [n.strip() for n in query.replace("vs", ",").split(",") if n.strip()]
    
    # Lấy metadata cho mỗi sản phẩm
    products = []
    for name in names:
        docs = retriever.get_relevant_documents(name)
        if docs:
            meta = docs[0].metadata
            products.append({
                "name": meta.get("Product Name", name),
                "price": meta.get("Current Price", "Unknown"),
                "basic_specs": meta.get("Basic Specs", "N/A"),
                "battery": meta.get("Battery Summary", "N/A"),
                "display": meta.get("Camera & Display Summary", "N/A"),
                "design": meta.get("Design Summary", "N/A"),
                "connectivity": meta.get("Connectivity Summary", "N/A")
            })
        else:
            products.append({
                "name": name,
                "price": "Not found",
                "basic_specs": "-",
                "battery": "-",
                "display": "-",
                "design": "-",
                "connectivity": "-"
            })

    # Nếu ít hơn 2 sản phẩm, báo lỗi
    if len(products) < 2:
        return "Please provide at least two products to compare."

    # Tạo bảng Markdown
    headers = "| Feature | " + " | ".join([p["name"] for p in products]) + " |"
    separator = "|" + "---|" * (len(products) + 1)
    rows = [
        "| Price | " + " | ".join([p["price"] for p in products]) + " |",
        "| Basic Specs | " + " | ".join([p["basic_specs"] for p in products]) + " |",
        "| Battery | " + " | ".join([p["battery"] for p in products]) + " |",
        "| Display | " + " | ".join([p["display"] for p in products]) + " |",
        "| Design | " + " | ".join([p["design"] for p in products]) + " |",
        "| Connectivity | " + " | ".join([p["connectivity"] for p in products]) + " |",
    ]

    return "\n".join([headers, separator] + rows)


compare_products_tool = Tool(
    name="compare_products",
    func=compare_products,
    description=(
        "Only Use this tool to when user mention to compare multiple smartphones (2 or more) by their names. "
        "Input should be a string of product names separated by commas or 'vs'. "
        "Example: 'iPhone 15 Pro, Samsung Galaxy S24' or 'iPhone 15 Pro vs Samsung Galaxy S24'."
        "When user mention compare, must return markdown to the final answer"
    ),
)


# Danh sách tools
#tools = [retriever_tool, compare_products_tool, product_search_tool, promotion_tool]
tools = [retriever_tool, promotion_tool, compare_products_tool]

def get_num_tokens_from_messages(self, messages):
    text = "\n".join([m.content for m in messages])
    return len(tokenizer.encode(text))

# Patch method vào ChatTogether class (not instance)
ChatTogether.get_num_tokens_from_messages = get_num_tokens_from_messages

# --- Khởi tạo ConversationSummaryBufferMemory ---
memory = ConversationSummaryBufferMemory(
    llm=llm,  # bắt buộc truyền llm để nó tóm tắt khi vượt giới hạn
    max_token_limit=1024,  # giới hạn token cho lịch sử
    return_messages=True,
    output_key="output",
)

#OPENAI_FUNCTIONS, ZERO_SHOT_REACT_DESCRIPTION
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True, 
    return_intermediate_steps=True,
)

def get_agent():
    return agent


