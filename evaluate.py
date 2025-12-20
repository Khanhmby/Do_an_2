import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import pickle

# --- 1. CẤU HÌNH ---
data_dir = r'F:\Desktop\Do_an_2\datasets\AugmentedAlzheimerDataset'
weights_dir = r'F:\Desktop\Do_an_2\weights'

cnn_model_path = os.path.join(weights_dir, 'simple_cnn_best.keras')
resnet_model_path = os.path.join(weights_dir, 'resnet50_finetuned_best.keras')
class_names_path = os.path.join(weights_dir, 'class_names.pkl')

IMG_HEIGHT = 170
IMG_WIDTH = 170
BATCH_SIZE = 32

# --- 2. HÀM NẠP DỮ LIỆU ĐÃ SỬA LỖI ---
def load_data_and_convert_to_numpy():
    print("--- Đang nạp dữ liệu kiểm thử (Validation set) ---")
    
    #  shuffle=True và seed=123 để khớp với lúc Train
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True 
    )
    
    # Lấy tên lớp
    class_names = val_ds.class_names
    print(f"Các lớp: {class_names}")

    print("Đang chuyển đổi dữ liệu sang Numpy Array để cố định thứ tự...")
    X_val = []
    y_val = []

    # Lặp qua dataset để lấy hết dữ liệu ra
    for images, labels in val_ds:
        X_val.append(images.numpy())
        y_val.append(labels.numpy())

    # Gộp các batch lại thành 1 mảng lớn
    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    
    print(f"-> Tổng số ảnh test: {len(y_val)}")
    return X_val, y_val, class_names

# --- 3. HÀM ĐÁNH GIÁ ---
def evaluate_model(model_path, model_name, X_test, y_test, class_names):
    print(f"\n--- Đang đánh giá: {model_name} ---")
    
    if not os.path.exists(model_path):
        print(f"⚠️ Không tìm thấy file: {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Lỗi nạp model: {e}")
        return None

    # Dự đoán
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Tính chỉ số
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"-> Accuracy: {acc:.4f}")
    
    # Vẽ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {acc:.2%}')
    plt.ylabel('Thực tế (True Label)')
    plt.xlabel('Dự đoán (Predicted Label)')
    plt.tight_layout()
    plt.show()
    
    return {'name': model_name, 'acc': acc, 'prec': p, 'rec': r, 'f1': f1}

# --- 4. MAIN ---
def main():
    # 1. Lấy dữ liệu (Dạng Numpy để đảm bảo không bị xáo trộn lại)
    try:
        X_test, y_true, class_names = load_data_and_convert_to_numpy()
    except Exception as e:
        print(f"Lỗi nạp dữ liệu: {e}")
        return

    results = []
    
    # 2. Chạy đánh giá
    res_cnn = evaluate_model(cnn_model_path, "Simple CNN", X_test, y_true, class_names)
    if res_cnn: results.append(res_cnn)
    
    res_resnet = evaluate_model(resnet_model_path, "ResNet50", X_test, y_true, class_names)
    if res_resnet: results.append(res_resnet)
    
    # 3. Vẽ biểu đồ so sánh cột
    if len(results) == 2:
        print("\n--- Vẽ biểu đồ so sánh ---")
        metrics = ['acc', 'prec', 'rec', 'f1']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, [results[0][m] for m in metrics], width, label='Simple CNN', color='#4e79a7')
        rects2 = ax.bar(x + width/2, [results[1][m] for m in metrics], width, label='ResNet50', color='#e15759')
        
        ax.set_ylabel('Scores')
        ax.set_title('So sánh hiệu năng mô hình')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend()
        ax.set_ylim(0, 1.15)
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')
                            
        autolabel(rects1)
        autolabel(rects2)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()