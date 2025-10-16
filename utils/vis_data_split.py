import os
import glob
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import json


def create_balanced_visibility_split(
        root_dir,
        train_dir,
        val_dir,
        test_dir,
        train_ratio=0.75,
        val_ratio=0.15,
        test_ratio=0.10,
        random_state=42,
        min_samples_per_class_test=50  # 确保测试集每类至少50个样本
):
    """
    专门为交通能见度数据集设计的三分割策略，处理严重的类别不平衡问题
    
    参数:
    root_dir (str): 包含类别子文件夹的原始数据根目录
    train_dir (str): 训练集保存目录
    val_dir (str): 验证集保存目录  
    test_dir (str): 测试集保存目录
    train_ratio (float): 训练集比例
    val_ratio (float): 验证集比例
    test_ratio (float): 测试集比例
    random_state (int): 随机种子
    min_samples_per_class_test (int): 测试集每类最少样本数
    """

    # 验证比例总和
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("训练、验证、测试集比例之和必须等于1.0")

    # 创建输出目录
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 收集所有图片路径和标签
    all_image_paths = []
    all_labels = []
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    # 统计每个类别的样本数
    class_sample_counts = {}
    for class_name in class_names:
        class_path = os.path.join(root_dir, class_name)
        
        image_extensions = ["*.jpg", "*.jpeg", "*.png"]  # 只使用小写模式
        class_images = []
        for ext_pattern in image_extensions:
            class_images.extend(glob.glob(os.path.join(class_path, ext_pattern)))

        # 去重，防止万一有重复
        class_images = list(set(class_images))

        all_image_paths.extend(class_images)
        all_labels.extend([class_name] * len(class_images))
        class_sample_counts[class_name] = len(class_images)
        print(f"类别 '{class_name}': {len(class_images)} 张图片")

    # 执行分层三分割
    # 首先分离出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(all_image_paths, all_labels, test_size=test_ratio, random_state=random_state, stratify=all_labels)

    # 然后从剩余数据中分离训练集和验证集
    # 重新计算训练集和验证集的比例
    remaining_ratio = train_ratio + val_ratio
    adjusted_train_ratio = train_ratio / remaining_ratio

    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, train_size=adjusted_train_ratio, random_state=random_state, stratify=y_temp)

    # 统计每个类别在各个数据集中的分布
    train_class_counts = Counter(y_train)
    val_class_counts = Counter(y_val)
    test_class_counts = Counter(y_test)

    # 创建统计信息DataFrame
    stats_data = []
    for class_name in sorted(class_names):
        original_count = class_sample_counts[class_name]
        train_count = train_class_counts.get(class_name, 0)
        val_count = val_class_counts.get(class_name, 0)
        test_count = test_class_counts.get(class_name, 0)
        
        # 计算比例
        train_ratio_actual = train_count / original_count if original_count > 0 else 0
        val_ratio_actual = val_count / original_count if original_count > 0 else 0
        test_ratio_actual = test_count / original_count if original_count > 0 else 0
        
        stats_data.append({
            '类别名称': class_name,
            '原始总数': original_count,
            '训练集数量': train_count,
            '验证集数量': val_count,
            '测试集数量': test_count,
            '训练集比例': f"{train_ratio_actual:.3f}",
            '验证集比例': f"{val_ratio_actual:.3f}",
            '测试集比例': f"{test_ratio_actual:.3f}"})
    
    # 创建DataFrame
    stats_df = pd.DataFrame(stats_data)
    
    # 添加总计行
    total_original = stats_df['原始总数'].sum()
    total_train = stats_df['训练集数量'].sum()
    total_val = stats_df['验证集数量'].sum()
    total_test = stats_df['测试集数量'].sum()
    
    total_row = pd.DataFrame({
        '类别名称': ['总计'],
        '原始总数': [total_original],
        '训练集数量': [total_train],
        '验证集数量': [total_val],
        '测试集数量': [total_test],
        '训练集比例': [f"{total_train/total_original:.3f}"],
        '验证集比例': [f"{total_val/total_original:.3f}"],
        '测试集比例': [f"{total_test/total_original:.3f}"]
    })
    
    stats_df = pd.concat([stats_df, total_row], ignore_index=True)

    # 复制文件到目标目录
    def copy_files_with_structure(file_paths, file_labels, destination_dir):
        for file_path, label in zip(file_paths, file_labels):
            # 创建类别子目录
            target_class_dir = os.path.join(destination_dir, label)
            os.makedirs(target_class_dir, exist_ok=True)

            # 复制文件
            filename = os.path.basename(file_path)
            target_file_path = os.path.join(target_class_dir, filename)
            shutil.copy2(file_path, target_file_path)

    # 执行文件复制
    copy_files_with_structure(X_train, y_train, train_dir)
    copy_files_with_structure(X_val, y_val, val_dir)
    copy_files_with_structure(X_test, y_test, test_dir)
    
    return stats_df


if __name__ == "__main__":
    SOURCE_ROOT_DIR = r'C:\Users\mjynj\Desktop\vis_recognize\img_data\data_anhui'
    data_root = os.path.join(os.path.dirname(__file__), "..", "data")
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    stats_df = create_balanced_visibility_split(SOURCE_ROOT_DIR, 
                                                train_dir, 
                                                val_dir, 
                                                test_dir, 
                                                train_ratio=0.75, 
                                                val_ratio=0.15, 
                                                test_ratio=0.10, 
                                                random_state=6666)
    print(stats_df)
