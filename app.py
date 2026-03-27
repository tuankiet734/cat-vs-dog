import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Cấu hình trang Web (Phải nằm ở dòng đầu tiên của code Streamlit)
st.set_page_config(page_title="Cat vs Dog Detector", page_icon="🐾")

st.title("🐾 Cat vs Dog Detector")
st.write("### Project Web Deploy - Phát hiện Mèo & Chó")
st.write("**Sinh viên thực hiện:** Nguyễn Đông Phương - 2286400025")
st.divider()

# 2. Load Model (Dùng cache để không bị load lại gây chậm)
@st.cache_resource
def load_model():
    # Sử dụng MobileNetV2 nhẹ và nhanh, phù hợp cho Web Cloud
    model = MobileNetV2(weights='imagenet')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Lỗi tải model: {e}")

# 3. Giao diện Upload ảnh
uploaded_file = st.file_uploader("📤 Chọn một bức ảnh chó hoặc mèo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Hiển thị ảnh người dùng tải lên
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption='Ảnh đã tải lên', use_container_width=True)
    
    # Nút bấm dự đoán
    if st.button('🔍 Dự đoán ngay', use_container_width=True):
        with st.spinner('Máy đang phân tích hình ảnh...'):
            # 4. Tiền xử lý ảnh
            # Đảm bảo ảnh ở hệ màu RGB và resize về 224x224
            img = image_data.convert('RGB')
            img = img.resize((224, 224))
            
            # Chuyển thành array và thêm chiều batch
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            # Chuẩn hóa theo chuẩn MobileNetV2
            x = preprocess_input(x)

            # 5. Dự đoán
            preds = model.predict(x)
            # Lấy top 3 kết quả dự đoán từ ImageNet
            decoded_preds = decode_predictions(preds, top=3)[0]
            
            top_label = decoded_preds[0][1] # Tên giống loài (class name)
            prob = decoded_preds[0][2]      # Xác suất (0 -> 1)

            # 6. Logic kiểm tra Chó hay Mèo dựa trên từ khóa
            cat_keywords = ['cat', 'tabby', 'tiger', 'siamese', 'persian', 'lynx', 'leopard', 'kitten']
            dog_keywords = ['dog', 'terrier', 'retriever', 'spaniel', 'shepherd', 'husky', 'corgi', 'pug', 'poodle', 'beagle']

            check_str = top_label.lower()
            is_cat = any(k in check_str for k in cat_keywords)
            is_dog = any(k in check_str for k in dog_keywords)
            
            # 7. Hiển thị kết quả ra màn hình
            st.subheader("Kết quả phân tích:")
            
            if is_dog:
                st.success(f"🐶 **ĐÂY LÀ CHÓ!**")
                st.info(f"Giống loài chi tiết: **{top_label.replace('_', ' ').title()}**")
            elif is_cat:
                st.success(f"🐱 **ĐÂY LÀ MÈO!**")
                st.info(f"Giống loài chi tiết: **{top_label.replace('_', ' ').title()}**")
            else:
                st.warning(f"🤔 Máy đoán đây là: **{top_label.replace('_', ' ').title()}** (Có thể không phải chó/mèo nhà)")

            # Hiển thị độ tin cậy
            st.write(f"Độ tin cậy: **{prob*100:.2f}%**")
            st.progress(float(prob))
