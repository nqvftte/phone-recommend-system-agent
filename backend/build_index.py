import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Đường dẫn đến file CSV chứa dữ liệu sản phẩm
DATA_PATH = "data/products_summarized.csv"

# Load dữ liệu từ CSV
df = pd.read_csv(DATA_PATH)

# Định nghĩa embedding model
embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")

# Tạo chuỗi văn bản để nhúng (semantic search) – không dùng Promotion
texts = (
    df["Product Name"].fillna("") + " | " +
    "Basic Specs: " + df["Basic Specs"].fillna("") + " | " +
    df["Battery Summary"].fillna("") + " | " +
    df["Connectivity Summary"].fillna("") + " | " +
    df["Design Summary"].fillna("") + " | " +
    df["Camera & Display Summary"].fillna("") + " | " +
    "Price: " + df["Current Price"].astype(str)
).tolist()

# Metadata để lưu thông tin hiển thị và lọc
metadatas = df[[
    "Product Name",
    "Current Price",
    "Old Price",
    "Product URL",
    "Promotion",  # vẫn giữ trong metadata để hiển thị nếu cần
    "Basic Specs",
    "Camera & Display",
    "Battery & Charging",
    "Connectivity",
    "Design & Material",
    "Color Variants",
    "Battery Summary",
    "Connectivity Summary",
    "Design Summary",
    "Camera & Display Summary"
]].to_dict("records")

# Tạo FAISS vectorstore
vectorstore = FAISS.from_texts(texts, embedding, metadatas=metadatas)

# Lưu index ra đĩa
os.makedirs("data/faiss_index", exist_ok=True)
vectorstore.save_local("data/faiss_index")

print("✅ FAISS index built and saved to data/faiss_index")
