import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from datasets.vis_dataset import VisibilityDataset
from utils import config


def img_dataloader(data_dir_path):
    '''
    从已按类别分子文件夹的目录中加载图像路径和标签
    '''
    all_img_paths = []
    all_labels = []
    expected_labels_str = [str(i) for i in range(config.NUM_CLASSES)]
    label_to_idx = {label_str: idx for idx, label_str in enumerate(expected_labels_str)}

    if not os.path.isdir(data_dir_path):
        return [], []

    for label_str in expected_labels_str:
        class_dir = os.path.join(data_dir_path, label_str)
        if not os.path.isdir(class_dir):
            continue

        img_patterns = ["*.jpg", "*.jpeg", "*.png"]
        for pattern in img_patterns:
            for img_path in glob.glob(os.path.join(class_dir, pattern)):
                all_img_paths.append(img_path)
                all_labels.append(label_to_idx[label_str])

    return all_img_paths, all_labels


def create_weighted_sampler(labels, balance_factor=config.BALANCE_FACTOR):
    '''
    创建加权采样器来处理不平衡数据集，支持可调节的平衡程度。
    
    Args:
        labels: 训练标签列表
        balance_factor: 平衡因子，控制采样策略的激进程度
                       - 0.0: 使用原始分布（不进行平衡）
                       - 0.5: 中等平衡（推荐）
                       - 1.0: 完全平衡（激进策略）
    '''
    class_counts = Counter(labels)
    total_samples = len(labels)

    # 计算每个类别的权重
    class_weights = {}
    for class_idx in range(config.NUM_CLASSES):
        if class_idx in class_counts:
            epsilon = 1e-5
            # 计算完全平衡的权重
            balanced_weight = total_samples / (config.NUM_CLASSES * (class_counts[class_idx] + epsilon))
            # 原始权重（1.0）
            original_weight = 1.0
            
            # 使用线性插值混合原始分布和平衡分布
            class_weights[class_idx] = (1 - balance_factor) * original_weight + balance_factor * balanced_weight
        else:
            class_weights[class_idx] = 1.0

    # 为每个样本分配权重
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=config.USE_SAMPLER_REPLACEMENT)


def collate_fn_filter_none(batch):
    '''
    过滤掉None的批次数据，支持特征和标签输出
    '''
    # 过滤掉None的样本
    batch = [item for item in batch if item is not None and len(item) == 3]
    if not batch:
        return None, None, None

    # 分离原始图像、增强图像和标签
    original_images = [item[0] for item in batch]
    augmented_images = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    # 使用默认的collate函数处理每个部分
    original_batch = torch.stack(original_images)
    augmented_batch = torch.stack(augmented_images)
    label_batch = torch.tensor(labels)

    return original_batch, augmented_batch, label_batch


def worker_init_fn():
    '''
    DataLoader worker初始化函数，确保每个worker有不同但确定的随机种子
    '''
    # 获取主进程的随机种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(train_dir, val_dir, augment=config.USE_AUGMENTATION, weighted_sampler=True):
    '''
    获取训练和验证数据加载器
    weighted_sampler：使用加权采样器
    '''
    # 根据现有结构，获取图像和对应路径，构建dataset
    train_img_paths, train_labels = img_dataloader(train_dir)
    val_img_paths, val_labels = img_dataloader(val_dir)
    train_dataset = VisibilityDataset(train_img_paths, train_labels, augment, is_train=True)
    val_dataset = VisibilityDataset(val_img_paths, val_labels, False, is_train=False)

    # 创建采样器
    if weighted_sampler:
        train_sampler = create_weighted_sampler(train_labels, balance_factor=config.BALANCE_FACTOR)
        train_loader = DataLoader(train_dataset, 
                                  batch_size=config.BATCH_SIZE, 
                                  sampler=train_sampler,
                                  num_workers=0,
                                  pin_memory=True,
                                  collate_fn=collate_fn_filter_none,
                                  worker_init_fn=worker_init_fn)
    else:
        train_loader = DataLoader(train_dataset, 
                                  batch_size=config.BATCH_SIZE, 
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  collate_fn=collate_fn_filter_none,
                                  worker_init_fn=worker_init_fn)

    val_loader = DataLoader(val_dataset, 
                            batch_size=config.BATCH_SIZE, 
                            shuffle=False,
                            num_workers=0,  # 增加多进程支持
                            pin_memory=True,  # 启用内存固定
                            collate_fn=collate_fn_filter_none,
                            worker_init_fn=worker_init_fn)

    return train_loader, val_loader, train_labels


if __name__ == "__main__":
    train_loader, val_loader, train_labels = get_dataloader(config.TRAIN_DATA_ROOT, config.VAL_DATA_ROOT, config.USE_AUGMENTATION, weighted_sampler=True)
    for i, (original_image, augmented_image, label) in enumerate(train_loader):
        print(original_image.shape, augmented_image.shape, label.shape)
        break