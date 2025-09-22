import os
from llama_index.llms.together import TogetherLLM
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import ReActAgent
from retriever2 import load_hybrid_retriever
from dotenv import load_dotenv

# Nạp biến môi trường
load_dotenv()

# Đảm bảo API key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
retriever = load_hybrid_retriever()

llm = TogetherLLM(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
) #You should change your model name to be matched with NVIDIA_API_KEY


# Test query
# results = retriever.retrieve("Tôi cần Điện thoại OPPO Find N3 Flip 5G 12GB/256GB Hồng")
# for r in results:
#     print(r.node.get_text())  # Xem text của kết quả

query_engine = RetrieverQueryEngine.from_args(
    llm=llm,
    retriever=retriever
)

# 2. Tạo tool cho smartphone recommendation
phone_recommend_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="smartphone_recommender",
    description=(
        "Answers user questions and recommends smartphones based on preferences "
        "(e.g., gaming, photography, battery life, budget). "
        "Can compare models, provide specs, prices, promotions, and purchase links. "
        "Always retrieve information from the tool without hallucinating."
    )
)

# 3. Định nghĩa agent với system prompt bằng tiếng Anh
Smartphone_Recommendation_Agent = ReActAgent(
    llm=llm,
    tools=[phone_recommend_tool],
    verbose=False,
    system_prompt=(
        "You are a helpful AI assistant specializing in recommending smartphones. "
        "Use the provided tool to retrieve and recommend the most suitable phones "
        "based on user needs (gaming, photography, long battery, budget, etc.). "
        "Always base your answers on the retrieved data and avoid making up products. "
        "Provide clear, concise recommendations with reasoning."
    )
)
