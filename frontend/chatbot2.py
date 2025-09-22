import streamlit as st
import requests
import uuid
from streamlit_chat import message  # pip install streamlit-chat

# Cấu hình
API_URL = "http://localhost:8000"  # Đổi thành backend API thật
st.set_page_config(page_title="Phone Recommender", page_icon="📱", layout="centered")

# Sinh user_id duy nhất cho session
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())

# Lưu lịch sử hội thoại
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of (role, text)

st.title("📱 Phone Recommender Chatbot")

def call_agent(user_message):
    """Gửi câu hỏi đến API và lấy phản hồi."""
    try:
        resp = requests.post(
            f"{API_URL}/chat",
            json={"user_id": st.session_state["user_id"], "message": user_message},
            timeout=20
        )
        if resp.status_code == 200:
            return resp.json().get("response", "Không nhận được phản hồi.")
        return f"Lỗi API ({resp.status_code})"
    except Exception as e:
        return f"Lỗi kết nối: {str(e)}"

# Chat input
user_input = st.chat_input("Nhập câu hỏi của bạn...")
if user_input:
    # Thêm tin nhắn user
    st.session_state["history"].append(("user", user_input))
    # Gọi agent
    reply = call_agent(user_input)
    st.session_state["history"].append(("bot", reply))

# Nút reset hội thoại
if st.button("🔄 Reset hội thoại"):
    try:
        requests.post(f"{API_URL}/reset", json={"user_id": st.session_state["user_id"]})
    except:
        pass
    st.session_state["history"] = []
    st.success("Hội thoại đã được reset!")

# Hiển thị lịch sử chat
for i, (role, msg) in enumerate(st.session_state["history"]):
    if role == "user":
        message(msg, is_user=True, key=f"user_{i}")
    else:
        message(msg, is_user=False, key=f"bot_{i}")
