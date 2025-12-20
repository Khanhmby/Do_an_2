import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import tempfile

# --- CẤU HÌNH ---
IMG_HEIGHT = 170
IMG_WIDTH = 170
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="AI Chẩn đoán Alzheimer",
    page_icon="🧠",
    layout="wide"
)

# --- CSS GIAO DIỆN ---
st.markdown("""
    <style>
    .result-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        height: 3em;
        font-weight: bold;
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Hệ thống Chẩn đoán Alzheimer qua ảnh MRI")
st.markdown("---")

# 2. HÀM LOAD MODEL THỦ CÔNG
def load_manual_model(file_path):
    #Load toàn bộ model từ file .keras hoặc .h5
    tf.keras.backend.clear_session()
    try:
        # Load model trực tiếp (bao gồm cả kiến trúc và trọng số)
        model = tf.keras.models.load_model(file_path)
        return model, "Thành công"
    except Exception as e:
        return None, f"Lỗi load model: {str(e)}"

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# 3. SIDEBAR - CẤU HÌNH THỦ CÔNG
st.sidebar.header("⚙️ Cấu hình Model")

# 1. Chọn loại Model
selected_model_type = st.sidebar.selectbox(
    "1. Chọn kiến trúc Model:",
    ["Simple CNN", "ResNet50"],
    help="Chọn đúng kiến trúc tương ứng với file trọng số bạn đã train."
)

# 2. Upload file
uploaded_model_file = st.sidebar.file_uploader(
    "2. Tải lên file Trọng số (.h5):",
    type=["h5", "keras", "weights.h5"]
)

if uploaded_model_file:
    file_mb = uploaded_model_file.size / (1024 * 1024)
    st.sidebar.success(f"File: {uploaded_model_file.name} ({file_mb:.1f} MB)")

confidence_threshold = st.sidebar.slider("Ngưỡng tin cậy:", 0, 100, 60)

# 4. GIAO DIỆN CHÍNH
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Tải ảnh MRI")
    uploaded_image = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Ảnh MRI gốc", use_container_width=True)
    else:
        st.info("Vui lòng tải ảnh lên.")

with col2:
    st.subheader("📊 Kết quả Chẩn đoán")
    
    if uploaded_image and uploaded_model_file:
        if st.button("🔍 CHẨN ĐOÁN NGAY"):
            with st.spinner('Đang tải model và phân tích...'):
                
                # Lưu file tạm
                with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
                    tmp_file.write(uploaded_model_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Gọi hàm load
                model, status = load_manual_model(tmp_path)
                
                # Xóa file tạm
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                
                if model is None:
                    st.error("❌ LỖI LOAD MODEL!")
                    st.error(status)
                    st.warning("Gợi ý: Kiểm tra xem bạn chọn 'Simple CNN' nhưng lại upload file 'ResNet' (hoặc ngược lại) không?")
                else:
                    # Dự đoán
                    processed_img = preprocess_image(image)
                    predictions = model.predict(processed_img)
                    
                    pred_idx = np.argmax(predictions[0])
                    pred_label = CLASS_NAMES[pred_idx]
                    confidence = 100 * np.max(predictions[0])
                    
                    # Hiển thị kết quả
                    color = "#28a745" if "Non" in pred_label else "#dc3545"
                    
                    st.markdown(f"""
                        <div class="result-card" style="background-color: {color}; color: white;">
                            <h3 style="margin:0;">Kết quả dự đoán</h3>
                            <h1 style="font-size: 3em; margin: 10px 0;">{pred_label}</h1>
                            <p>Độ tin cậy: <strong>{confidence:.2f}%</strong></p>
                            <p style="font-size: 0.8em;">(Model: {selected_model_type})</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if confidence < confidence_threshold:
                        st.warning("⚠️ Độ tin cậy thấp.")
                        
                    st.markdown("#### Chi tiết xác suất:")
                    for i, class_name in enumerate(CLASS_NAMES):
                        prob = predictions[0][i] * 100
                        st.progress(int(prob))
                        st.caption(f"{class_name}: {prob:.2f}%")

    elif not uploaded_model_file:
        st.write("👈 Vui lòng tải file trọng số và chọn loại model ở cột trái.")
    elif not uploaded_image:
        st.write("👈 Vui lòng tải ảnh MRI.")

# Footer
st.markdown("---")
st.caption("Manual Selection Mode.")