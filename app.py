import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# 1. Cấu hình
st.set_page_config(page_title="Pet Detector", page_icon="🐾")
st.title("🐾 Máy dò Mèo vs Chó")
st.write("Nguyễn Đông Phương - 2286400025")

# 2. Load Model an toàn
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

try:
    model = load_model()
except Exception as e:
    st.error("Đang khởi tạo hệ thống AI, vui lòng đợi trong giây lát...")
    st.stop()

# 3. Giao diện
uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Ảnh đã upload', use_container_width=True)
    
    if st.button('Dự đoán'):
        with st.spinner('Đang phân tích...'):
            # Tiền xử lý
            img_resized = img.resize((224, 224))
            x = np.array(img_resized)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Dự đoán
            preds = model.predict(x)
            label = decode_predictions(preds, top=1)[0][0]
            name = label[1].lower()
            prob = label[2]

            # Kiểm tra từ khóa
            is_dog = any(k in name for k in ['dog', 'puppy', 'terrier', 'retriever', 'husky', 'poodle'])
            is_cat = any(k in name for k in ['cat', 'tabby', 'kitten', 'siamese', 'persian'])

            st.divider()
            if is_dog:
                st.success(f"🐶 ĐÂY LÀ CHÓ! (Giống: {name.title()})")
            elif is_cat:
                st.success(f"🐱 ĐÂY LÀ MÈO! (Giống: {name.title()})")
            else:
                st.warning(f"Máy đoán là: {name.title()}")
            
            st.write(f"Độ chính xác: {prob*100:.2f}%")
            st.progress(float(prob))
