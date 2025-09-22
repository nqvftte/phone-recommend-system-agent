# data_loader.py
import pandas as pd
from llama_index.core import Document

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
