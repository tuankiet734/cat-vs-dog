import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Cấu hình trang Web
st.set_page_config(page_title="Cat vs Dog Detector", page_icon="🐾")

st.title("🐾 Cat vs Dog Detector")
st.write("Project Web Deploy - Detect Cat & Dog")
st.write("thành kiet phương")

# 2. Load Model
@st.cache_resource
def load_model():
    # Sử dụng MobileNetV2 đã train sẵn trên ImageNet
    model = MobileNetV2(weights='imagenet')
    return model

with st.spinner('Đang tải model... vui lòng chờ chút nhé!'):
    model = load_model()

# 3. Giao diện Upload ảnh
uploaded_file = st.file_uploader("Chọn ảnh chó hoặc mèo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption='Ảnh đã upload', use_container_width=True)
    
    if st.button('Dự đoán ngay'):
        with st.spinner('Đang phân tích...'):
            img = image_data.convert('RGB')
            img = img.resize((224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)
            decoded_preds = decode_predictions(preds, top=3)[0]
            
            top_label = decoded_preds[0][1]
            prob = decoded_preds[0][2]

            cat_keywords = ['cat', 'tabby', 'tiger', 'siamese', 'persian', 'lynx', 'leopard', 'kitten']
            dog_keywords = ['dog', 'terrier', 'retriever', 'spaniel', 'shepherd', 'husky', 'corgi', 'pug', 'poodle']

            check_str = top_label.lower()
            is_cat = any(k in check_str for k in cat_keywords)
            is_dog = any(k in check_str for k in dog_keywords)
            
            st.divider()
            if is_dog:
                st.success(f"Kết quả: ĐÂY LÀ CHÓ (DOG) - ({top_label.replace('_', ' ').title()})")
                st.progress(float(prob))
                st.write(f"Độ tin cậy: {prob*100:.2f}%")
            elif is_cat:
                st.success(f"Kết quả: ĐÂY LÀ MÈO (CAT) - ({top_label.replace('_', ' ').title()})")
                st.progress(float(prob))
                st.write(f"Độ tin cậy: {prob*100:.2f}%")
            else:
                st.warning(f"Máy dự đoán là: {top_label.title()} (Không chắc chắn là chó hay mèo)")
