# build_index.py
import pandas as pd
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Đọc CSV và tạo Document list
def load_product_docs_csv(path):
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        text = (
            f"Tên sản phẩm: {row['Tên sản phẩm']}\n"
            f"Giá hiện tại: {row['Giá hiện tại']}\n"
            f"Giá cũ: {row['Giá cũ']}\n"
            f"Màu sắc: {row['Màu sắc']} (Hex: {row['Hex']})\n"
            f"Khuyến mãi: {row['Khuyến mãi']}\n"
            f"Cấu hình: {row['Cấu hình (rút gọn)']}\n"
            f"Link: {row['Link']}\n"
            f"Thumbnail: {row['Thumbnail']}"
        )
        docs.append(Document(
            text=text,
            metadata={
                "product_name": row['Tên sản phẩm'],
                "price": row['Giá hiện tại'],
                "old_price": row['Giá cũ'],
                "color": row['Màu sắc'],
                "config": row['Cấu hình (rút gọn)'],
                "link": row['Link'],
                "thumbnail": row['Thumbnail']
            }
        ))
    return docs

if __name__ == "__main__":
    # 1. Load product docs
    docs = load_product_docs_csv("./data/products.csv")

    # 2. Chọn embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3"  # hoặc "BAAI/bge-m3" cho đa ngôn ngữ
    )

    # 3. Build VectorStoreIndex và lưu lại
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    index.storage_context.persist("./data/storage")
    print("Index built and saved to ./data/storage")
