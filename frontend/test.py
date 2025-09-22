from streamlit_chat import message
import streamlit as st

# Ví dụ dữ liệu bạn nhận được từ agent
response_text = "Here is the phone you might like: Samsung Galaxy Z Fold6"
image_url = "https://img.tgdd.vn/imgt/f_webp,fit_outside,quality_75,s_100x100/https://cdn.tgdd.vn/Products/Images/42/324994/tcl-406s-blue-1-1-180x125.jpg"

# Hiển thị tin nhắn người dùng
message("I want a phone with big screen![]({image_url}", is_user=True, key="user_msg_1")

# Hiển thị tin nhắn của bot
message(response_text, is_user=False, key="bot_msg_1")

# Hiển thị ảnh bên dưới tin nhắn (nên dùng st.image và thêm key nếu cần)
# Tạo hai cột
col1, col2, col3 = st.columns(3)

# Đặt ảnh vào từng cột
with col1:
    st.image(image_url, width=200, caption="Samsung Galaxy Z Fold6")

with col2:
    st.image(image_url, width=200, caption="Samsung Galaxy Z Fold6")

with col3:
    st.image("https://cdnv2.tgdd.vn/mwg-static/tgdd/Products/Images/42/330028/viettel-sumo-4g-t2-xanh-1-638633149523802257-180x125.jpg", width=200, caption="Samsung Galaxy Z Fold6")
# Một tin nhắn người dùng khác
message("I want a phone with big screen[Điện thoại realme Note 60 6GB/128GB](https://www.thegioididong.com/dtdd/realme-note-60)", is_user=True, key="user_msg_2")
