from llama_index.llms.together import TogetherLLM
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool

# 1. Tạo query engine từ retriever (retriever load từ retriever.py)
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

# 4. Hàm query: nhận input tiếng Việt, agent xử lý tiếng Anh, dịch kết quả sang tiếng Việt
def recommend_smartphone(user_query):
    # Gọi agent (agent hiểu prompt tiếng Anh)
    english_response = Smartphone_Recommendation_Agent.run(user_query)

    # Dịch sang tiếng Việt
    translation_prompt = f"Translate the following text into natural Vietnamese:\n\n{english_response}"
    vietnamese_answer = llm.complete(translation_prompt).text

    return vietnamese_answer


# --- Ví dụ chạy ---
if __name__ == "__main__":
    question = "Tôi cần một điện thoại pin trâu, chụp ảnh đẹp và giá tầm 8 triệu"
    answer = recommend_smartphone(question)
    print(answer)
