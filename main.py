import os

# --- 1. CẤU HÌNH HỆ THỐNG (QUAN TRỌNG) ---
# Tắt OneDNN để tránh lỗi "could not create a memory object"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle

data_dir = r'F:\Desktop\Do_an_2\datasets\AugmentedAlzheimerDataset'
weights_dir = r'F:\Desktop\Do_an_2\weights'

old_weights_path = os.path.join(weights_dir, 'best_alzheimer_model.weights.h5')

new_model_path_best = os.path.join(weights_dir, 'simple_cnn_best.keras')

IMG_HEIGHT = 170
IMG_WIDTH = 170
BATCH_SIZE = 32

if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

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

# Lưu danh sách lớp
with open(os.path.join(weights_dir, 'class_names.pkl'), 'wb') as f:
    pickle.dump(class_names, f)

# Tối ưu hiệu năng
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# XÂY DỰNG MODEL (SIMPLE CNN) ---
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),
    
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),
    
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), 
    layers.Dense(num_classes, activation='softmax')
])

if os.path.exists(old_weights_path):
    print(f"\n Tìm thấy file trọng số cũ: {old_weights_path}")
    print("Đang nạp để train tiếp...")
    try:
        # load_weights cho Simple CNN thường ít lỗi hơn ResNet
        model.load_weights(old_weights_path, skip_mismatch=True)
        print("-> Nạp thành công!")
    except Exception as e:
        print(f"-> Lỗi nạp trọng số: {e}")
        print("-> Sẽ train lại từ đầu.")
else:
    print(f"\n Không tìm thấy file {old_weights_path}, sẽ train mới hoàn toàn.")

# Compile sau khi load weights
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    # 1. Lưu model tốt nhất (Full .keras)
    ModelCheckpoint(
        new_model_path_best, 
        save_best_only=True, 
        monitor='val_accuracy', 
        mode='max', 
        verbose=1
    ),
    
    # 2. Lưu backup sau MỖI Epoch (đề phòng tắt máy)
    ModelCheckpoint(
        filepath=os.path.join(weights_dir, 'simple_cnn_epoch_{epoch:02d}.keras'),
        save_best_only=False,
        save_freq='epoch',
        verbose=1
    ),
    
    # 3. Các chức năng hỗ trợ train
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]

#HUẤN LUYỆN
print("\n--- Bắt đầu Training ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50, 
    callbacks=callbacks
)

#VẼ BIỂU ĐỒ
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
plt.title('Accuracy (Simple CNN)')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss (Simple CNN)')
plt.show()

model.save(os.path.join(weights_dir, 'simple_cnn_final.keras'), include_optimizer=False)
print("\n Đã lưu file cuối cùng: simple_cnn_final.keras")