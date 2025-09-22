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
            result = resp.json()
            return {
                "response": result.get("response", "Không nhận được phản hồi."),
                "images": result.get("image_data", [])  # danh sách ảnh
            }
        return {"response": f"Lỗi API ({resp.status_code})", "images": []}
    except Exception as e:
        return {"response": f"Lỗi kết nối: {str(e)}", "images": []}

# Chat input
user_input = st.chat_input("Nhập câu hỏi của bạn...")
if user_input:
    # Thêm tin nhắn user
    st.session_state["history"].append(("user", user_input))

    # Gọi agent
    result = call_agent(user_input)
    reply = result["response"]
    images = result["images"]

    # Lưu phản hồi bot cùng hình ảnh
    st.session_state["history"].append(("bot", reply, images))


# Nút reset hội thoại
if st.button("🔄 Reset hội thoại"):
    try:
        requests.post(f"{API_URL}/reset", json={"user_id": st.session_state["user_id"]})
    except:
        pass
    st.session_state["history"] = []
    st.success("Hội thoại đã được reset!")

# Hiển thị lịch sử chat
for i, entry in enumerate(st.session_state["history"]):
    if entry[0] == "user":
        message(entry[1], is_user=True, key=f"user_{i}")
    else:
        # Hiển thị tin nhắn
        message(entry[1], is_user=False, key=f"bot_{i}")

        # Nếu có ảnh thì hiện trong các cột
        if len(entry) > 2 and entry[2]:
            cols = st.columns(len(entry[2]))
            for col, item in zip(cols, entry[2]):
                if item.get("image_link"):
                    with col:
                        st.image(item["image_link"], width=200, caption=item.get("name", ""))
