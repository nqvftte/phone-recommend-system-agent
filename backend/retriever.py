# retriever.py
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load dữ liệu sản phẩm
# df = pd.read_csv("data/products.csv")

# Embedding model (BGE-M3)
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},  # hoặc "cuda" nếu có GPU
    encode_kwargs={"normalize_embeddings": True}
)

# Load FAISS index (được build bằng build_index.py)
vectorstore = FAISS.load_local(
    "data/faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

# Tạo retriever (top 3 sản phẩm gần nhất)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
