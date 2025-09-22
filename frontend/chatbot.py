# frontend.py
import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000"  # Backend API

# Sinh user_id duy nhất cho session
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())

st.title("Phone Recommender Chatbot")

# Hiển thị lịch sử hội thoại
if "history" not in st.session_state:
    st.session_state["history"] = []

# Form nhập câu hỏi
query = st.text_input("Nhập câu hỏi của bạn:")

col1, col2 = st.columns([1,1])
with col1:
    send = st.button("Gửi")
with col2:
    reset = st.button("Reset hội thoại")

# Gửi câu hỏi đến API
if send and query.strip():
    resp = requests.post(
        f"{API_URL}/chat",
        json={"user_id": st.session_state["user_id"], "message": query}
    )
    print("Agent response:", resp)

    reply = resp.json().get("response", "Không nhận được phản hồi.")
    st.session_state["history"].append(("Bạn", query))
    st.session_state["history"].append(("Bot", reply))
    query = ""

# Reset hội thoại
if reset:
    requests.post(f"{API_URL}/reset", json={"user_id": st.session_state["user_id"]})
    st.session_state["history"] = []
    st.success("Đã reset hội thoại!")

# Hiển thị hội thoại
st.subheader("Lịch sử chat:")
for role, msg in st.session_state["history"]:
    if role == "Bạn":
        st.markdown(f"**{role}:** {msg}")
    else:
        st.markdown(f"**{role}:** {msg}")
