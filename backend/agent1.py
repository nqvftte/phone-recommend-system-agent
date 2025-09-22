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

# def format_product(doc):
#     meta = doc.metadata
#     name = meta.get("Tên sản phẩm", "Sản phẩm không rõ")
#     link = meta.get("Link", "#")
#     price = meta.get("Giá hiện tại", "Không rõ")
#     promo = meta.get("Khuyến mãi", "Không có")
#     color = meta.get("Màu sắc", "Không rõ")
#     specs = meta.get("Cấu hình (rút gọn)", "Không rõ")
#     thumbnail = meta.get("Thumbnail", "")

#     # Nếu muốn hiển thị hình ảnh luôn (Markdown)
#     thumb_md = f"![Ảnh]({thumbnail})" if thumbnail else ""

#     return (
#         f"### [{name}]({link})\n"
#         f"- Giá: {price}\n"
#         f"- Màu sắc: {color}\n"
#         f"- Cấu hình: {specs}\n"
#         f"- Khuyến mãi: {promo}\n"
#         f"{thumb_md}"
#     )


# def retrieve_products(query: str) -> str:
#     docs = retriever.get_relevant_documents(query)
#     results = []
#     for doc in docs:
#         meta = doc.metadata
#         name = meta.get("Tên sản phẩm", "Sản phẩm")
#         price = meta.get("Giá hiện tại", "Không rõ")
#         link = meta.get("Link", "#")
#         results.append(f"- [{name}]({link}) — Giá: {price}")
#     return "\n".join(results) if results else "Không tìm thấy sản phẩm phù hợp."

# retriever_tool = Tool(
#     name="product_retriever",
#     func=retrieve_products,
#     description="Dùng để tìm smartphone theo yêu cầu (theo tên, giá, cấu hình)."
# )
def format_product(doc):
    meta = doc.metadata
    name = meta.get("Tên sản phẩm", "Sản phẩm")
    link = meta.get("Link", "#")
    price = meta.get("Giá hiện tại", "Không rõ")
    color = meta.get("Màu sắc", "Không rõ")
    specs = meta.get("Cấu hình (rút gọn)", "Không rõ")
    

    return f"- [{name}]({link}) — Giá: {price}\n  Màu: {color}\n  Cấu hình: {specs}"

def search_products_formatted(input_str: str) -> str:
    parts = [p.strip() for p in input_str.split("|")]
    query = parts[0] if parts else ""
    color = parts[1].lower() if len(parts) > 1 else None

    docs = retriever.get_relevant_documents(query)
    results = []

    for doc in docs:
        meta_color = str(doc.metadata.get("Màu sắc", "") or "").strip().lower()
        if color and meta_color != color:
            continue
        results.append(format_product(doc))

    return "\n\n".join(results) if results else "Không tìm thấy sản phẩm phù hợp."

retriever_tool = Tool(
    name="product_retriever",
    func=search_products_formatted,
    description="Dùng để tìm smartphone theo yêu cầu (theo tên, giá, cấu hình, màu sắc). Input: 'query' hoặc 'query|Màu sắc'."
)

#check_promotion
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


#Tool gọi màu 
from langchain.agents import Tool

# def search_products_formatted(input_str: str) -> str:
#     """
#     Tìm điện thoại theo từ khoá và màu (tùy chọn).
#     Input: "query|color" hoặc chỉ "query".
#     Output: Chuỗi danh sách sản phẩm.
#     """
#     parts = input_str.split("|")
#     query = parts[0].strip()
#     color = parts[1].strip() if len(parts) > 1 else None

#     docs = retriever.get_relevant_documents(query)
#     results = []
#     for doc in docs:
#         meta = doc.metadata
#         if color and meta.get("Màu sắc", "").strip().lower() != color.lower():
#             continue
#         name = meta.get("Tên sản phẩm")
#         link = meta.get("Link")
#         price = meta.get("Giá hiện tại", "Không rõ")
#         color_name = meta.get("Màu sắc", "Không rõ")
#         results.append(f"- [{name}]({link}) — Giá: {price} — Màu: {color_name}")

#     if not results:
#         return "Không tìm thấy sản phẩm phù hợp."
#     return "\n".join(results)
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
# prompt = ChatPromptTemplate.from_messages([
#     ("system", 
#      "Bạn là trợ lý tư vấn smartphone. Luôn trả lời bằng tiếng Việt. "
#      "Trước khi trả lời người dùng, bạn BẮT BUỘC phải gọi tool `retriever_tool` để tìm thông tin sản phẩm. Không được tự tạo câu trả lời nếu không có kết quả từ công cụ."
#      "Chỉ trả lời dựa trên kết quả từ tool. "
#      "Nếu công cụ không trả về gì, hãy xin lỗi và nói rằng không tìm thấy sản phẩm phù hợp."),
#     ("placeholder", "{chat_history}"),
#     ("human", "{input}"),
#     ("placeholder", "{agent_scratchpad}"),
# ])
# system_prompt = """
# Bạn là trợ lý tư vấn smartphone. 
# LUÔN trả lời hoàn toàn bằng tiếng Việt, không được dùng tiếng Anh.
# Nếu người dùng hỏi về cấu hình, giá, màu sắc, bạn phải tìm thông tin từ retriever trước.
# Nếu cần gọi tool để lấy dữ liệu, hãy gọi tool và sau đó trả lời bằng tiếng Việt dựa trên dữ liệu.
# """
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "Bạn là trợ lý tư vấn smartphone. Thought bằng tiếng Việt, LUÔN trả lời bằng tiếng Việt, không dùng tiếng Anh."),
#     ("placeholder", "{chat_history}"),
#     ("human", "{input}"),
#     ("placeholder", "{agent_scratchpad}"),
# ])
# system_prompt = """
# Bạn là AI Agent tư vấn điện thoại.
# Luôn suy nghĩ (thought) và lập luận bằng tiếng Việt trong scratchpad.
# Nếu cần hiển thị reasoning, hãy dùng định dạng:
# "Suy nghĩ: ..."
# "Trả lời: ..."
# """


# Memory (tóm tắt để tiết kiệm token)
# memory = ConversationSummaryBufferMemory(
#     memory_key="chat_history",
#     return_messages=True,
#     llm=llm,
#     max_token_limit=2000
# )

# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True
# )

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

# Custom prompt that forces Vietnamese output
# template = """Bạn là một trợ lý AI chuyên tư vấn và tìm điện thoại.
# Hãy luôn trả lời bằng tiếng Việt, và kèm thông tin giá, cấu hình, khuyến mãi (nếu có).
# Sử dụng các công cụ khi cần thiết.

# Câu hỏi: {input}
# {agent_scratchpad}"""

# prompt = PromptTemplate(
#     template=template,
#     input_variables=["input", "agent_scratchpad"]
# )
#OPENAI_FUNCTIONS, ZERO_SHOT_REACT_DESCRIPTION
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True, 
)
system_prompt = (
    "Bạn là Trợ lý AI của cửa hàng điện thoại QV, chuyên tư vấn smartphone.\n\n"
    "Quy tắc bắt buộc khi trả lời:\n"
    "Khi thấy tên điện thoại thì trả lời 3 tên điện thoại đầu tiên mà retrieve được"
    "LUÔN tuân thủ định dạng sau khi suy nghĩ:"
    "Thought: [suy nghĩ]"
    "Action: [tên công cụ cần gọi]"
    "Action Input: [dữ liệu đầu vào]"
    "Nếu đã nhận dữ liệu từ công cụ, PHẢI tạo Final Answer: ngay, không chỉ dừng ở Thought."
    "Ví dụ:"

    "Người dùng: Có điện thoại Xiaomi Redmi Note 14 không?"
    "Thought: Tôi cần tìm trong dữ liệu."
    "Action: product_retriever"
    "Action Input: Xiaomi Redmi Note 14"

    "Observation: (kết quả sản phẩm)"
    "Thought: Tôi đã tìm thấy sản phẩm."
    "Final Answer: Có, chúng tôi có bán Xiaomi Redmi Note 14 (8GB/256GB, các màu: Đen, Xanh lá, Tím)."
    "1. LUÔN trả lời bằng tiếng Việt, văn phong ngắn gọn, tự nhiên.\n"
    "2. LUÔN sử dụng công cụ `product_retriever` để tìm thông tin, không tự suy đoán.\n"
    "3. Chỉ tư vấn về smartphone trong cơ sở dữ liệu FAISS index. KHÔNG bịa tên hoặc thông tin sản phẩm.\n"
    "4. Không thay đổi tên sản phẩm khi trả lời.\n"
    "5. Nếu người dùng hỏi:\n"
    "   - Màu sắc (ví dụ: 'có màu đen không?') → Trả lời Có/Không và liệt kê chính xác các màu có trong dữ liệu.\n"
    "   - Giá, cấu hình, phiên bản → Trả lời ngắn gọn, dựa đúng dữ liệu.\n"
    "6. Nếu không tìm thấy thông tin phù hợp → trả lời: 'Xin lỗi, tôi không tìm thấy dữ liệu phù hợp.'\n"
    "7. Không trả lời về sản phẩm ngoài danh sách dữ liệu.\n"
    "8. Không trả lời các câu hỏi không liên quan đến smartphone.\n\n"
    "Mục tiêu: Hỗ trợ người dùng chọn điện thoại chính xác dựa trên dữ liệu có sẵn."
)
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
])
# agent.agent.llm_chain.prompt.template = prompt
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

