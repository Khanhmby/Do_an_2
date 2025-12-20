import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path


DATASET_PATH = r"F:\Desktop\Do_an_2\datasets\AugmentedAlzheimerDataset"

def load_dataset_metadata(data_path):
    """
    Quét thư mục và tạo DataFrame chứa đường dẫn ảnh và nhãn.
    """
    image_paths = []
    labels = []
    
    # Lấy danh sách các lớp (tên thư mục con)
    classes = os.listdir(data_path)
    
    print(f"🔄 Đang quét dữ liệu từ: {data_path}...")
    
    for class_name in classes:
        class_dir = os.path.join(data_path, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Lấy tất cả file ảnh (jpg, png, jpeg)
        for img_type in ["*.jpg", "*.jpeg", "*.png"]:
            files = list(Path(class_dir).rglob(img_type))
            for file in files:
                image_paths.append(str(file))
                labels.append(class_name)
                
    df = pd.DataFrame({'path': image_paths, 'label': labels})
    print(f"✅ Đã tìm thấy {len(df)} ảnh thuộc {len(df['label'].unique())} lớp.")
    return df

def plot_class_distribution(df):
    """
    Vẽ biểu đồ cột thể hiện số lượng ảnh trong mỗi lớp.
    Giúp phát hiện vấn đề mất cân bằng dữ liệu (Class Imbalance).
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    ax = sns.countplot(x='label', data=df, palette='viridis', order=df['label'].value_counts().index)
    
    plt.title('Phân bố số lượng ảnh giữa các lớp', fontsize=15)
    plt.xlabel('Mức độ sa sút trí tuệ', fontsize=12)
    plt.ylabel('Số lượng ảnh', fontsize=12)
    plt.xticks(rotation=15)
    
    # Hiển thị số lượng cụ thể trên đầu cột
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.show()

def plot_sample_images(df, num_samples=5):
    """
    Hiển thị ngẫu nhiên một số ảnh mẫu từ mỗi lớp để kiểm tra trực quan.
    """
    unique_labels = df['label'].unique()
    
    fig, axes = plt.subplots(len(unique_labels), num_samples, figsize=(15, 3 * len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        # Lấy ngẫu nhiên num_samples ảnh từ lớp hiện tại
        sample_df = df[df['label'] == label].sample(num_samples)
        
        for j, (_, row) in enumerate(sample_df.iterrows()):
            img_path = row['path']
            try:
                # Đọc ảnh grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if len(unique_labels) == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                
                ax.imshow(img, cmap='bone') # cmap='bone' rất tốt cho ảnh X-ray/MRI
                ax.axis('off')
                
                if j == 0:
                    ax.set_title(label, fontsize=12, fontweight='bold', loc='left')
            except Exception as e:
                print(f"Lỗi đọc file {img_path}: {e}")
                
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def plot_pixel_intensity_distribution(df, samples_per_class=100):
    """
    Vẽ biểu đồ phân phối cường độ điểm ảnh (Pixel Intensity).
    Giúp quyết định cách Normalization (ví dụ chia cho 255 hay dùng Mean/Std).
    """
    plt.figure(figsize=(12, 6))
    
    unique_labels = df['label'].unique()
    
    for label in unique_labels:
        # Lấy mẫu để tính toán cho nhanh (thay vì toàn bộ dataset)
        sample_paths = df[df['label'] == label].sample(min(samples_per_class, len(df[df['label']==label])))['path']
        
        pixel_values = []
        for path in sample_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                pixel_values.extend(img.flatten())
        
        # Vẽ KDE plot (Kernel Density Estimate)
        sns.kdeplot(pixel_values, label=label, fill=True, alpha=0.3)
        
    plt.title('Phân phối cường độ điểm ảnh (Pixel Intensity)', fontsize=15)
    plt.xlabel('Giá trị Pixel (0-255)', fontsize=12)
    plt.ylabel('Mật độ', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# --- HÀM MAIN ---
if __name__ == "__main__":
    # Kiểm tra đường dẫn tồn tại không
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Lỗi: Không tìm thấy thư mục '{DATASET_PATH}'.")
        print("👉 Vui lòng sửa biến DATASET_PATH trong code trỏ đúng đến thư mục chứa 4 folder con.")
        # Tạo dữ liệu giả lập để demo nếu không tìm thấy folder thật
        print("⚠️ Đang tạo dữ liệu giả lập để demo code...")
        data = {
            'path': ['fake_path.jpg'] * 400,
            'label': ['NonDemented']*100 + ['VeryMildDemented']*100 + ['MildDemented']*100 + ['ModerateDemented']*100
        }
        df = pd.DataFrame(data)
        # Chỉ chạy plot distribution vì không có ảnh thật
        plot_class_distribution(df)
    else:
        # 1. Load dữ liệu
        df = load_dataset_metadata(DATASET_PATH)
        
        if not df.empty:
            # 2. Vẽ biểu đồ phân bố lớp
            plot_class_distribution(df)
            
            # 3. Vẽ ảnh mẫu
            plot_sample_images(df)
            
            # 4. Vẽ phân phối pixel
            plot_pixel_intensity_distribution(df)