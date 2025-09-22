# agent.py
import os
from langchain.tools import Tool, tool, StructuredTool
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
from langchain.schema import SystemMessage
from langchain.chains import LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor, LLMSingleActionAgent
from langchain.agents.mrkl.output_parser import MRKLOutputParser
from retriever import retriever  # retriever đã load FAISS hoặc vectorstore
from langchain.schema import StrOutputParser
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from dotenv import load_dotenv

# Nạp biến môi trường
load_dotenv()

# Đảm bảo API key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Khởi tạo LLM (Together)
# llm = ChatTogether(
#     model="meta-llama/Llama-3-8b-chat-hf",
#     temperature=0.7,
#     max_tokens=512,  # tránh vượt quá giới hạn token
# )


# --- Tạo tokenizer cho LLaMA ---
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

from langchain.tools import Tool

# Hàm format hiển thị sản phẩm
def format_product(doc):
    meta = doc.metadata
    name = meta.get("Tên sản phẩm", "Sản phẩm")
    link = meta.get("Link", "#")
    price = meta.get("Giá hiện tại", "Không rõ")
    color = meta.get("Màu sắc", "Không rõ")
    specs = meta.get("Cấu hình (rút gọn)", "Không rõ")
    return f"- [{name}]({link}) — Giá: {price}\n  Màu: {color}\n  Cấu hình: {specs}"


# Tìm sản phẩm (basic)
def search_products(input_str: str, filter_color=False) -> str:
    parts = [p.strip() for p in input_str.split("|")]
    query = parts[0] if parts else ""
    color = parts[1].lower() if len(parts) > 1 else None

    docs = retriever.get_relevant_documents(query)
    results = []
    for doc in docs:
        meta_color = str(doc.metadata.get("Màu sắc", "") or "").strip().lower()
        if filter_color and color and meta_color != color:
            continue
        results.append(format_product(doc))

    return "\n\n".join(results) if results else "Không tìm thấy sản phẩm phù hợp."


# Tool: tìm sản phẩm (dùng cho agent, lọc màu nếu có)
## Định nghĩa schema input
class ProductRetrieverInput(BaseModel):
    query: str

def search_products_formatted(query: str) -> str:
    parts = [p.strip() for p in query.split("|")]
    q = parts[0]
    color = parts[1].lower() if len(parts) > 1 else None

    docs = retriever.get_relevant_documents(q)
    results = []
    for doc in docs:
        meta_color = str(doc.metadata.get("Màu sắc", "") or "").strip().lower()
        if color and meta_color != color:
            continue
        results.append(format_product(doc))

    return "\n\n".join(results) if results else "Không tìm thấy sản phẩm phù hợp."

# Tạo tool với schema chuẩn
retriever_tool = StructuredTool.from_function(
    func=search_products_formatted,
    name="product_retriever",
    description="Dùng để tìm smartphone theo yêu cầu (theo tên, giá, cấu hình, màu sắc). Input: 'query' hoặc 'query|Màu sắc'.",
    args_schema=ProductRetrieverInput
)



# Tool: kiểm tra khuyến mãi
def check_promotion(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "Không tìm thấy sản phẩm nào phù hợp với yêu cầu của bạn."

    results = []
    for doc in docs:
        meta = doc.metadata
        name = meta.get("Tên sản phẩm", "Sản phẩm")
        promo = meta.get("Khuyến mãi", "")
        if promo and promo.lower() != "không có":
            results.append(f"- {name}: {promo}")
        else:
            results.append(f"- {name}: Hiện tại không có khuyến mãi.")

    return "\n".join(results)


promotion_tool = Tool(
    name="promotion_tool",
    func=check_promotion,
    description="Kiểm tra sản phẩm có khuyến mãi hay không. Input: tên sản phẩm hoặc từ khóa tìm kiếm."
)


# Tool: so sánh sản phẩm
def compare_products(query: str) -> str:
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
    description="Dùng để so sánh nhiều sản phẩm (2 hoặc hơn) dựa trên tên. Input là chuỗi chứa tên sản phẩm, phân tách bởi dấu phẩy hoặc 'vs'."
)


# Tool: tìm sản phẩm (danh sách, không lọc màu bắt buộc)
product_search_tool = Tool(
    name="product_search_tool",
    func=lambda q: search_products(q, filter_color=False),
    description="Tìm điện thoại theo yêu cầu. Input: 'query|color' hoặc 'query'. Kết quả trả về danh sách sản phẩm."
)


# Danh sách tools cho agent
tools = [retriever_tool, compare_products_tool, product_search_tool, promotion_tool]

# Prompt cho agent
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "Bạn là trợ lý tư vấn sản phẩm thông minh của cửa hàng điện thoại QV, trả lời LUÔN bằng tiếng Việt. Khi cần, bạn có thể dùng công cụ tìm kiếm để trả lời chính xác."
#      "LUÔN sử dụng công cụ (product_retriever), không thay đổi tên sản phẩm"
#      "Chỉ tư vấn về smartphone, không bịa sản phẩm ngoài database."
#      "Chỉ trả lời dựa trên dữ liệu FAISS index."
#      "KHÔNG tự bịa tên hoặc thông tin sản phẩm."
#      "Nếu người dùng hỏi về:"
#     "- Màu sắc (ví dụ: có màu đen không?) → Trả lời dạng Có/Không, liệt kê các màu chính xác có trong dữ liệu."
#     "- Các thông tin khác (giá, cấu hình) → Trả lời ngắn gọn, chỉ dựa trên dữ liệu."
#     "Nếu không có bất kỳ thông tin nào, nói: Xin lỗi, tôi không tìm thấy dữ liệu phù hợp."),
#     ("placeholder", "{chat_history}"),
#     ("human", "{input}"),
#     ("placeholder", "{agent_scratchpad}"),
# ])

# Patch method vào ChatTogether class (not instance)
# ChatTogether.get_num_tokens_from_messages = get_num_tokens_from_messages

# --- Khởi tạo ConversationSummaryBufferMemory ---
# memory = MemorySaver()

# # 4. Tạo agent với prompt tiếng Việt
# agent = create_react_agent(
#     model=llm,
#     tools=tools,
#     checkpointer=memory,
#     messages_modifier=lambda messages: [
#         {
#             "role": "system",
#             "content": (
#                 "Bạn là một trợ lý AI chuyên tư vấn và tìm điện thoại. "
#                 "Luôn trả lời bằng tiếng Việt, kèm thông tin giá, cấu hình, và khuyến mãi (nếu có). "
#                 "Hãy dùng các công cụ để lấy dữ liệu chính xác."
#             ),
#         },
#         *messages,
#     ],
# )

# # 5. Gọi agent (giữ history cho từng user)
# config = {"configurable": {"thread_id": "user-1"}}

# for event in agent.stream(
#     {"messages": [{"role": "user", "content": "Tìm điện thoại chơi game dưới 10 triệu"}]},
#     config,
# ):
#     if "agent" in event:
#         print(event["agent"]["messages"][-1]["content"])


# def get_agent():
#     return agent


def get_agent():
    llm = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        temperature=0.7,
        max_tokens=512
    )
    memory = MemorySaver()

    # Tạo prompt tiếng Việt
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "Bạn là trợ lý AI. Khi người dùng hỏi về điện thoại, bạn PHẢI:\n"
        "- Nghĩ (Thought) xem cần làm gì\n"
        "- Gọi đúng công cụ (Action) với đúng input (Action Input)\n"
        "- Không tự bịa thông tin nếu không có dữ liệu\n"
        "- Chỉ trả lời Final Answer sau khi nhận Observation từ tool.\n"
        "Bạn PHẢI tuân thủ đúng format: Thought → Action → Action Input → Observation → Thought → Final Answer."),
        
        # "Few-shot examples"
        ("human", "Ví dụ: Tìm điện thoại Samsung màu đen."),
        ("ai", "Thought: Tôi cần tìm sản phẩm Samsung màu đen.\n"
            "Action: product_retriever\n"
            "Action Input: Samsung|đen"),
        
        ("human", "Ví dụ: So sánh iPhone 13 và Samsung Galaxy A52."),
        ("ai", "Thought: Tôi cần so sánh 2 sản phẩm.\n"
            "Action: compare_products\n"
            "Action Input: iPhone 13, Samsung Galaxy A52"),
        
        # Placeholder cho input thực tế của người dùng
        ("human", "{messages}")
    ])

    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        prompt=prompt  # giờ là ChatPromptTemplate, không phải list
    )
    return agent






