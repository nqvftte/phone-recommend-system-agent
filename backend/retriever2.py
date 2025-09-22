# retriever.py
import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from data_loader import load_product_docs_csv  # hoặc copy hàm nếu không muốn import
from dotenv import load_dotenv

# Nạp biến môi trường
load_dotenv()

# Đảm bảo API key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Tạo LLM (ví dụ OpenAI)
llm = TogetherLLM(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
) #You should change your model name to be matched with NVIDIA_API_KEY
docs = load_product_docs_csv("./data/products.csv")
storage_context = StorageContext.from_defaults(persist_dir="./data/storage")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")  # hoặc model bạn build
index = load_index_from_storage(storage_context, embed_model=embed_model)

def load_hybrid_retriever():

    # 2. Tạo vector retriever
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

    # 3. Load docs để BM25 hoạt động
    
    bm25_retriever = BM25Retriever.from_defaults(nodes=docs, similarity_top_k=3)

    # 4. Kết hợp retrievers
    retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        llm=llm,
        num_queries=1,
        use_async=True,
    )

    return retriever
