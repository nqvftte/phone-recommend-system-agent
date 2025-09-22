import os
from dotenv import load_dotenv

# Nạp biến môi trường
load_dotenv()

import os
from langchain_together import Together
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from retriever import retriever

# API key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

#meta-llama/Llama-3.1-8B-Instruct-Turbo --- meta-llama/Llama-3.3-70B-Instruct-Turbo-Free ---- meta-llama/Llama-3-8b-chat-hf

# LLM cấu hình
llm = Together(
    model="meta-llama/Llama-3-8b-chat-hf",
    # temperature=0.7,
    # max_tokens=256,  # Giảm token sinh ra để tránh vượt limit
    together_api_key=TOGETHER_API_KEY,
)

# Memory với auto-summary (giữ context dài nhưng không overflow)
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1500,    # Tóm tắt nếu context > 3000 tokens
    memory_key="chat_history",
    return_messages=True
)

# Tool tìm sản phẩm
def search_products(query: str):
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])

tools = [
    Tool(
        name="Search Products",
        func=search_products,
        description="Tìm điện thoại phù hợp theo yêu cầu người dùng"
    )
]

# Khởi tạo agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    memory=memory,
)

def get_agent():
    return agent
