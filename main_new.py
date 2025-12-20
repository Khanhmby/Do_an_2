import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# ==========================================
# 1. CẤU HÌNH & KIỂM TRA GPU
# ==========================================
# Kiểm tra xem có GPU không (Không bắt buộc, nhưng tốt nếu có)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ĐƯỜNG DẪN DỮ LIỆU (Sửa lại cho đúng máy bạn)
data_dir = r'F:\Desktop\Do_an_2\datasets\AugmentedAlzheimerDataset'

if not os.path.exists(data_dir):
    print(f"LỖI: Không tìm thấy thư mục: {data_dir}")
    exit()

BATCH_SIZE = 32
IMG_SIZE = (128, 128) # Resize về 128x128

# ==========================================
# 2. CHIA DỮ LIỆU (SPLIT 8:1:1)
# ==========================================
print("\n--- Đang chia dữ liệu theo tỷ lệ 8:1:1 ---")

# BƯỚC 1: Lấy 80% làm tập Train
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,   # Cắt 20% ra để riêng (làm Val + Test)
    subset="training",      # Lấy 80% còn lại làm Training
    seed=123,               # Seed cố định để kết quả không đổi
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

# BƯỚC 2: Lấy 20% còn lại (gọi tạm là val_and_test)
val_and_test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",    # Lấy 20% đã cắt ra ở trên
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

# BƯỚC 3: Chia đôi tập 20% này -> 10% Val và 10% Test
# Tính số lượng batch trong tập 20%
val_batches = tf.data.experimental.cardinality(val_and_test_ds)
test_batches = val_batches // 2 

# Chia tách
test_ds = val_and_test_ds.take(test_batches)
val_ds = val_and_test_ds.skip(test_batches)

print(f"Số batch Train: {tf.data.experimental.cardinality(train_ds)}")
print(f"Số batch Validation: {tf.data.experimental.cardinality(val_ds)}")
print(f"Số batch Test: {tf.data.experimental.cardinality(test_ds)}")

# Lấy tên Class
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes: {class_names}")

# Tối ưu hóa hiệu năng (Caching & Prefetching)
AUTOTUNE = tf.data.AUTOTUNE

# Train cần shuffle để học tốt hơn
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# Val và Test không cần shuffle, chỉ cần cache để chạy nhanh
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. XÂY DỰNG MÔ HÌNH (CUSTOM CNN IMPROVED)
# ==========================================
print("\n--- Đang khởi tạo mô hình ---")

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
], name="Augmentation")

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    # Pre-processing
    data_augmentation,
    layers.Rescaling(1./255),
    
    # Block 1
    layers.Conv2D(32, 3, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    
    # Block 2
    layers.Conv2D(64, 3, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    
    # Block 3
    layers.Conv2D(128, 3, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    
    # Block 4 (Deep features)
    layers.Conv2D(256, 3, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(),
    
    # Classifier
    layers.GlobalAveragePooling2D(), # Thay cho Flatten để nhẹ model
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# ==========================================
# 4. TRAINING VỚI CALLBACKS
# ==========================================
lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\n--- Bắt đầu Training ---")
history = model.fit(
    train_ds,
    validation_data=val_ds, # Chỉ dùng Val để kiểm tra lúc train
    epochs=10,
    callbacks=[early_stopping, lr_scheduler]
)

# ==========================================
# 5. ĐÁNH GIÁ (EVALUATION)
# ==========================================
# Vẽ biểu đồ Training
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# --- QUAN TRỌNG: Đánh giá trên tập TEST ---
# Đây là kết quả khách quan nhất để báo cáo đồ án
print("\n--- Đánh giá trên tập TEST (Dữ liệu chưa từng thấy) ---")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Độ chính xác trên tập Test: {test_acc*100:.2f}%")

# ==========================================
# 6. LƯU MODEL & CLASS NAMES
# ==========================================
if not os.path.exists('weights'):
    os.makedirs('weights')

model.save('weights/alzheimer_cnn_8_1_1.keras')
print("\nĐã lưu model thành công.")

with open('weights/class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)
print("Đã lưu danh sách nhãn lớp.")