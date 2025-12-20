import os

# --- 1. CẤU HÌNH HỆ THỐNG ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle

# --- 2. THIẾT LẬP ĐƯỜNG DẪN ---
data_dir = r'F:\Desktop\Do_an_2\datasets\AugmentedAlzheimerDataset'
weights_dir = r'F:\Desktop\Do_an_2\weights'

# Đường dẫn file model tốt nhất
model_path_best = os.path.join(weights_dir, 'resnet50_finetuned_best.keras')

IMG_HEIGHT = 170
IMG_WIDTH = 170
TARGET_SIZE = 224 # Kích thước tối ưu cho ResNet
BATCH_SIZE = 16   # Giảm batch size vì ResNet tốn VRAM hơn

if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

# --- 3. NẠP DỮ LIỆU ---
print("--- Đang nạp dữ liệu ---")
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Các lớp: {class_names}")

# Tối ưu hiệu năng nạp dữ liệu
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 4. XÂY DỰNG MODEL (KIẾN TRÚC MẠNH MẼ) ---
print("\n--- Xây dựng Model ResNet50 Fine-Tune ---")

# Lớp tăng cường dữ liệu (Chỉ hoạt động khi training)
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05), # Xoay rất nhẹ (5%)
    layers.RandomZoom(0.05),
])

inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Bước 1: Augmentation
x = data_augmentation(inputs)

# Bước 2: Resize lên 224x224 (ResNet thích size này)
x = layers.Resizing(TARGET_SIZE, TARGET_SIZE)(x)

# Bước 3: Preprocess chuẩn của ResNet (quan trọng!)
# Hàm này sẽ chuyển đổi pixel phù hợp với cách ResNet được train trên ImageNet
x = preprocess_input(x)

# Bước 4: Base Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(TARGET_SIZE, TARGET_SIZE, 3))
base_model.trainable = False # Ban đầu đóng băng

x = base_model(x, training=False) # training=False để khóa BatchNormalization layer

# Bước 5: Classification Head (Phần đầu ra mới)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x) # Chống overfitting
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs, name="ResNet50_Alzheimer_FineTuned")

# --- 5. GIAI ĐOẠN 1: WARM-UP (Huấn luyện nhẹ) ---
print("\n🔥 GIAI ĐOẠN 1: Train lớp đầu ra (Base đóng băng)")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train nhanh 10 epochs để các lớp Dense học được chút ít
history_warmup = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# --- 6. GIAI ĐOẠN 2: FINE-TUNING (Huấn luyện sâu) ---
print("\n❄️ GIAI ĐOẠN 2: Unfreeze & Fine-Tune (Quan trọng nhất)")

# Mở khóa base model
base_model.trainable = True

# ResNet50 có khoảng 175 layers.
# Ta sẽ đóng băng 140 lớp đầu (giữ lại khả năng nhận diện cạnh cơ bản)
# Chỉ train lại khoảng 30-40 lớp cuối (nhận diện đặc trưng trừu tượng của não)
fine_tune_at = 140

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"-> Đã mở khóa từ layer {fine_tune_at} trở đi.")

# QUAN TRỌNG: Compile lại với Learning Rate CỰC NHỎ
# Nếu để LR lớn (như 0.001), nó sẽ phá vỡ các trọng số tốt sẵn có.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(model_path_best, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
]

# Train tiếp 20-30 epochs nữa
total_epochs = 30
history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history_warmup.epoch[-1], # Tiếp nối số epoch cũ
    callbacks=callbacks
)

# --- 7. VẼ BIỂU ĐỒ TỔNG HỢP ---
print("\n--- Vẽ biểu đồ kết quả ---")

# Nối lịch sử huấn luyện của 2 giai đoạn
acc = history_warmup.history['accuracy'] + history_finetune.history['accuracy']
val_acc = history_warmup.history['val_accuracy'] + history_finetune.history['val_accuracy']
loss = history_warmup.history['loss'] + history_finetune.history['loss']
val_loss = history_warmup.history['val_loss'] + history_finetune.history['val_loss']

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
# Vẽ vạch ngăn cách 2 giai đoạn
plt.plot([9, 9], plt.ylim(), label='Bắt đầu Fine Tuning', linestyle='--', color='green')
plt.legend(loc='lower right')
plt.title('Accuracy: Warmup + FineTuning')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot([9, 9], plt.ylim(), label='Bắt đầu Fine Tuning', linestyle='--', color='green')
plt.legend(loc='upper right')
plt.title('Loss: Warmup + FineTuning')

plt.show()