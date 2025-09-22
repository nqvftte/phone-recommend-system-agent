# backend/memory_store.py
import os
import json
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

MEMORY_DIR = "data/memory"

if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)

def memory_file(user_id: str) -> str:
    """Trả về đường dẫn file memory của user"""
    return os.path.join(MEMORY_DIR, f"{user_id}.json")

def load_memory(user_id: str) -> ConversationBufferMemory:
    """Load memory từ file JSON, nếu chưa có thì tạo mới"""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    file_path = memory_file(user_id)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            for turn in history:
                if turn["role"] == "user":
                    memory.chat_memory.add_message(HumanMessage(content=turn["content"]))
                else:
                    memory.chat_memory.add_message(AIMessage(content=turn["content"]))
        except Exception as e:
            print(f"⚠️ Không load được memory cho {user_id}: {e}")
    return memory

def save_memory(user_id: str, memory: ConversationBufferMemory):
    """Lưu toàn bộ hội thoại ra file JSON"""
    history = []
    for msg in memory.chat_memory.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        history.append({"role": role, "content": msg.content})
    with open(memory_file(user_id), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def reset_memory(user_id: str):
    """Xoá file memory của user"""
    file_path = memory_file(user_id)
    if os.path.exists(file_path):
        os.remove(file_path)
