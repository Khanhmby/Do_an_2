import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# ==========================================
# 1. CẤU HÌNH
# ==========================================
data_dir = r'F:\Desktop\Do_an_2\datasets\AugmentedAlzheimerDataset'
IMG_HEIGHT = 170
IMG_WIDTH = 170
BATCH_SIZE = 32
WEIGHTS_PATH = 'weights\simple_cnn_best.keras' # Đường dẫn file trọng số CNN tốt nhất

# Kiểm tra đường dẫn dữ liệu
if not os.path.exists(data_dir):
    print(f"❌ LỖI: Không tìm thấy thư mục dữ liệu tại: {data_dir}")
    exit()

# ==========================================
# 2. LOAD TÊN LỚP (CLASS NAMES)
# ==========================================
# Cố gắng load từ file pickle, nếu không có thì dùng mặc định
if os.path.exists('weights/class_names.pkl'):
    with open('weights/class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
else:
    print("⚠️ Không tìm thấy class_names.pkl, sử dụng tên mặc định.")
    class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

num_classes = len(class_names)
print(f"✅ Các nhãn lớp: {class_names}")

# ==========================================
# 3. TẠO TẬP DỮ LIỆU KIỂM THỬ (TEST SET)
# ==========================================
print("\n--- Đang nạp dữ liệu kiểm thử ---")
# Lưu ý quan trọng: shuffle=True để đảm bảo lấy ngẫu nhiên 20% dữ liệu
# (tránh trường hợp chỉ lấy toàn bộ ảnh của lớp cuối cùng)
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True 
)

# ==========================================
# 4. XÂY DỰNG LẠI KIẾN TRÚC CNN
# ==========================================
def build_cnn_model():
    """
    Phải xây dựng giống hệt kiến trúc lúc train (có BatchNormalization)
    thì mới load được trọng số.
    """
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(), # Quan trọng
        
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(), # Quan trọng
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(), # Quan trọng
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5), 
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

print("\n--- Đang khởi tạo mô hình và load trọng số ---")
model = build_cnn_model()

# Kiểm tra file trọng số
if os.path.exists(WEIGHTS_PATH):
    try:
        model.load_weights(WEIGHTS_PATH)
        print(f"✅ Đã load trọng số thành công từ: {WEIGHTS_PATH}")
    except ValueError as e:
        print("\n❌ LỖI LỆCH KIẾN TRÚC MODEL!")
        print("File trọng số không khớp với cấu trúc code hiện tại.")
        print(f"Chi tiết lỗi: {e}")
        exit()
else:
    print(f"❌ LỖI: Không tìm thấy file trọng số tại {WEIGHTS_PATH}")
    print("Vui lòng chạy file train.py trước để tạo file này.")
    exit()

# ==========================================
# 5. DỰ ĐOÁN VÀ TÍNH TOÁN MA TRẬN
# ==========================================
print("\n--- Đang thực hiện dự đoán (Vui lòng đợi...) ---")

y_true_all = [] # Chứa nhãn thực tế
y_pred_all = [] # Chứa nhãn dự đoán

# Lặp qua từng batch dữ liệu
# Vì shuffle=True nên ta phải lấy nhãn và dự đoán đồng thời trong vòng lặp
for images, labels in test_ds:
    # 1. Lấy nhãn thực tế của batch này
    y_true_all.extend(labels.numpy())
    
    # 2. Dự đoán cho batch này
    predictions = model.predict(images, verbose=0)
    predicted_ids = np.argmax(predictions, axis=1) # Lấy index có xác suất cao nhất
    y_pred_all.extend(predicted_ids)

# Chuyển về dạng numpy array để xử lý
y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# ==========================================
# 6. VẼ CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(y_true_all, y_pred_all)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.title('Confusion Matrix - Simple CNN Model')
plt.ylabel('Nhãn Thực tế (True Label)')
plt.xlabel('Nhãn Dự đoán (Predicted Label)')
plt.show()

# ==========================================
# 7. IN BÁO CÁO CHI TIẾT
# ==========================================
print("\n" + "="*50)
print("BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT)")
print("="*50)
print(classification_report(y_true_all, y_pred_all, target_names=class_names))