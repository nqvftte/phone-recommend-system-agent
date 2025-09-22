import streamlit as st
from streamlit_chat import message

st.set_page_config(page_title="Phone Recommender Agent", page_icon="📱", layout="wide")

# --- CSS tuỳ chỉnh ---
st.markdown("""
    <style>
        .main {
            background-color: #f9fafb;
        }
        .chat-bubble {
            padding: 12px 18px;
            margin: 5px 0;
            border-radius: 12px;
            max-width: 70%;
        }
        .user-bubble {
            background-color: #007bff;
            color: white;
            margin-left: auto;
        }
        .bot-bubble {
            background-color: #e5e5ea;
            color: black;
            margin-right: auto;
        }
        .product-card {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 10px;
            margin: 10px;
            width: 220px;
            display: inline-block;
            vertical-align: top;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
        }
        .product-card img {
            width: 100%;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("logo.jpg", width=120)
st.sidebar.title("⚡ Phone Recommender")
st.sidebar.markdown("Tìm smartphone theo nhu cầu của bạn.")
brand = st.sidebar.selectbox("Chọn hãng", ["Tất cả", "Samsung", "Xiaomi", "iPhone"])
budget = st.sidebar.slider("Ngân sách (triệu VNĐ)", 2, 50, (5, 15))
if st.sidebar.button("Reset hội thoại"):
    st.session_state.messages = []

# --- Chat State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📱 Chatbot Tư vấn Điện thoại")

# --- Hiển thị hội thoại ---
for i, (sender, msg) in enumerate(st.session_state.messages):
    bubble_class = "user-bubble" if sender == "user" else "bot-bubble"
    st.markdown(f'<div class="chat-bubble {bubble_class}">{msg}</div>', unsafe_allow_html=True)

# --- Ô nhập tin nhắn ---
user_input = st.text_input("Nhập tin nhắn...", key="user_input")
if st.button("Gửi") and user_input:
    # Append tin nhắn người dùng
    st.session_state.messages.append(("user", user_input))
    
    # Gọi backend (giả lập)
    bot_reply = f"Bạn muốn tìm điện thoại trong khoảng {budget[0]} - {budget[1]} triệu của hãng {brand}. Tôi gợi ý một số mẫu..."
    st.session_state.messages.append(("bot", bot_reply))

    # Hiển thị sản phẩm mẫu
    st.markdown("### Sản phẩm gợi ý")
    for i in range(3):
        st.markdown(f"""
        <div class="product-card">
            <img src="https://via.placeholder.com/220x150.png?text=Phone+{i+1}" />
            <h4>Điện thoại {i+1}</h4>
            <p><b>Giá:</b> {5+i} triệu VNĐ</p>
            <a href="https://example.com" target="_blank">Mua ngay</a>
        </div>
        """, unsafe_allow_html=True)
