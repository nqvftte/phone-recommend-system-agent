import json
import os

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "data/user_memory.json")

# Load profile từ JSON
def load_user_profile(user_id: str) -> str:
    if not os.path.exists(MEMORY_PATH):
        return "Chưa có dữ liệu"
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        memory = json.load(f)
    return memory.get(user_id, "Chưa có dữ liệu")

# Cập nhật profile theo query mới
def update_user_profile(user_id: str, query: str):
    memory = {}
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            memory = json.load(f)

    prev = memory.get(user_id, "")
    memory[user_id] = prev + f"\n- {query}"

    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
