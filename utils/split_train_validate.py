import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter


def create_stratified_split_and_copy(root_dir, train_dir, val_dir, train_ratio=0.9, random_state=42):
    """
    对指定根目录下的数据进行分层抽样，将图片复制到新的训练和验证集目录，
    并保持原始的类别子文件夹结构。

    参数:
    root_dir (str): 包含类别子文件夹的原始数据根目录。
    train_dir (str): 保存训练集图片的目标目录。
    val_dir (str): 保存验证集图片的目标目录。
    train_ratio (float): 训练集所占的比例 (0到1之间)。
    random_state (int): 随机种子，用于保证划分结果可复现。
    """

    # 1. 检查原始数据根目录是否存在
    if not os.path.isdir(root_dir):
        print(f"错误：原始数据根目录 '{root_dir}' 不存在。")
        return

    # 2. 创建目标训练集和验证集目录 (如果不存在)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    print(f"训练集将保存到: {train_dir}")
    print(f"验证集将保存到: {val_dir}")

    # 3. 收集所有类别的图片路径和标签
    all_image_paths = []
    all_labels = []
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    if not class_names:
        print(f"错误：在 '{root_dir}' 下未找到任何子文件夹（类别）。")
        return

    print(f"找到类别: {class_names}")

    for class_name in class_names:
        class_path = os.path.join(root_dir, class_name)
        # 你可以根据需要匹配更多图片格式
        image_extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
        class_images_found_this_run = []
        for ext_pattern in image_extensions:
            class_images_found_this_run.extend(glob.glob(os.path.join(class_path, ext_pattern)))

        # 去重，以防一个文件被多个模式匹配 (例如 file.jpg 和 file.JPG 在不区分大小写系统上)
        # 或者，如果文件名本身就是唯一的，可以不用set
        # unique_class_images = sorted(list(set(class_images_found_this_run))) # 如果需要绝对去重
        unique_class_images = sorted(class_images_found_this_run)  # 如果glob模式设计得当，通常不需要set

        if not unique_class_images:
            print(f"  - 警告: 类别 '{class_name}' 中未找到图片文件。")
            continue

        all_image_paths.extend(unique_class_images)
        all_labels.extend([class_name] * len(unique_class_images))  # 标签就是文件夹名
        print(f"  - 类别 '{class_name}': 找到 {len(unique_class_images)} 张图片。")

    if not all_image_paths:
        print("错误：未能从任何类别中收集到图片文件。")
        return

    print(f"\n总共收集到 {len(all_image_paths)} 张图片。")

    # 4. 执行分层抽样
    # train_test_split 需要 X (特征) 和 y (标签)
    # 这里 X 就是 image_paths, y 就是 all_labels
    # val_ratio = 1.0 - train_ratio
    print(f"正在进行分层抽样... 训练集比例: {train_ratio*100:.1f}%, 验证集比例: {(1-train_ratio)*100:.1f}%")

    # 确保每个类别的样本数至少为2，否则stratify会出问题
    # 或者如果某个类别只有一个样本，它将无法被划分
    label_counts = Counter(all_labels)
    min_samples_for_split = 2  # train_test_split中 stratify 至少需要每个类别有两个样本

    valid_paths_for_split = []
    valid_labels_for_split = []
    skipped_labels = set()

    # 筛选出可以进行分层划分的数据
    # 将样本数不足 min_samples_for_split 的类别的所有样本都放入训练集（或根据需求处理）
    paths_to_force_train = []
    labels_to_force_train = []

    for class_name in class_names:
        if label_counts[class_name] < min_samples_for_split:
            skipped_labels.add(class_name)
            print(f"  - 警告: 类别 '{class_name}' 样本数 ({label_counts[class_name]}) 过少，无法进行分层抽样，将全部放入训练集。")
            # 将这些样本直接加入到待强制放入训练集的部分
            for i, label in enumerate(all_labels):
                if label == class_name:
                    paths_to_force_train.append(all_image_paths[i])
                    labels_to_force_train.append(all_labels[i])
        else:
            # 收集可以正常划分的样本
            for i, label in enumerate(all_labels):
                if label == class_name:
                    valid_paths_for_split.append(all_image_paths[i])
                    valid_labels_for_split.append(all_labels[i])

    train_final_paths = list(paths_to_force_train)  # 强制放入训练集的
    train_final_labels = list(labels_to_force_train)
    val_final_paths = []
    val_final_labels = []

    if valid_paths_for_split:  # 只有当有足够样本进行划分时才调用 train_test_split
        train_paths_split, val_paths_split, train_labels_split, val_labels_split = train_test_split(
            valid_paths_for_split,
            valid_labels_for_split,
            train_size=train_ratio,
            random_state=random_state,
            stratify=valid_labels_for_split  # 使用 valid_labels_for_split 进行分层
        )
        train_final_paths.extend(train_paths_split)
        train_final_labels.extend(train_labels_split)
        val_final_paths.extend(val_paths_split)
        val_final_labels.extend(val_labels_split)
    else:
        print("  - 没有足够样本的类别进行标准的分层抽样。")

    print(f"\n划分结果: 训练集 {len(train_final_paths)} 张, 验证集 {len(val_final_paths)} 张。")

    # 5. 复制文件到目标目录
    def copy_files(file_paths, destination_root_dir, dataset_name="数据集"):
        print(f"\n开始复制文件到 {dataset_name} 目录: {destination_root_dir}")
        copied_count = 0
        for original_path in file_paths:
            try:
                # 从原始路径中提取类别名（即倒数第二个路径部分）
                class_name = os.path.basename(os.path.dirname(original_path))
                # 构建目标类别子文件夹路径
                target_class_dir = os.path.join(destination_root_dir, class_name)
                os.makedirs(target_class_dir, exist_ok=True)  # 创建类别子文件夹

                # 构建目标文件路径
                filename = os.path.basename(original_path)
                target_file_path = os.path.join(target_class_dir, filename)

                # 复制文件
                shutil.copy2(original_path, target_file_path)  # copy2 会同时复制元数据
                copied_count += 1
            except Exception as e:
                print(f"  - 复制文件 '{original_path}' 到 '{target_file_path}' 时出错: {e}")
        print(f"成功复制 {copied_count} 个文件到 {dataset_name} 目录。")

    # 复制训练集文件
    copy_files(train_final_paths, train_dir, "训练集")
    # 复制验证集文件
    copy_files(val_final_paths, val_dir, "验证集")

    print("\n数据划分和复制完成！")


if __name__ == "__main__":
    # --- 用户配置 ---
    # 1. 指定包含类别子文件夹 (0, 1, 2, 3, 4) 的原始数据根目录路径
    SOURCE_ROOT_DIR = r"/work/highway_data"  # <<< 你的原始数据根目录

    # 2. 指定保存训练集的目标目录
    TRAIN_DATA_DIR = r"/work/highway_train_data"  # <<< 训练集保存目录

    # 3. 指定保存验证集的目标目录
    VALIDATE_DATA_DIR = r"/highway_validate_data"  # <<< 验证集保存目录

    # 4. (可选) 训练集比例和随机种子
    TRAIN_RATIO = 0.9
    RANDOM_SEED = 42  # 为了结果可复现

    # --- 执行函数 ---
    if not os.path.exists(SOURCE_ROOT_DIR):
        print(f"错误：指定的原始数据根目录 '{SOURCE_ROOT_DIR}' 不存在。")
    else:
        create_stratified_split_and_copy(SOURCE_ROOT_DIR, TRAIN_DATA_DIR, VALIDATE_DATA_DIR, train_ratio=TRAIN_RATIO, random_state=RANDOM_SEED)
