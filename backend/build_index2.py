import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os

DATA_PATH = "data/products.csv"

# Đọc dữ liệu
df = pd.read_csv(DATA_PATH)

# Chuẩn bị text (mỗi sản phẩm = 1 chunk)
texts = []
metadatas = []

for _, row in df.iterrows():
    text = f"""
Sản phẩm: {row['Tên sản phẩm']}
Giá: {row['Giá hiện tại']} (Giá cũ: {row['Giá cũ']})
Cấu hình: {row['Cấu hình (rút gọn)']}
Khuyến mãi: {row['Khuyến mãi']}
"""
    texts.append(text.strip())

    # Metadata (để lọc nhanh hoặc hiển thị link, màu sắc...)
    metadatas.append({
        "Tên sản phẩm": row["Tên sản phẩm"],
        "Giá hiện tại": row["Giá hiện tại"],
        "Giá cũ": row["Giá cũ"],
        "Màu sắc": row["Màu sắc"],
        "Hex": row["Hex"],
        "Link": row["Link"],
        "Thumbnail": row["Thumbnail"],
        "Cấu hình": row["Cấu hình (rút gọn)"]
    })

# Embedding
embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")

# Build FAISS index
vectorstore = FAISS.from_texts(texts, embedding, metadatas=metadatas)

# Lưu index
os.makedirs("data/faiss_index", exist_ok=True)
vectorstore.save_local("data/faiss_index")

print("✅ FAISS index built: mỗi sản phẩm = 1 chunk, saved at data/faiss_index")
