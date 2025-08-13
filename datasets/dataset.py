import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import torchvision.transforms as T
from PIL import Image
from utils import config

class InputResize:
    """
    自适应调整图像尺寸，支持任意输入尺寸，内部自动处理到指定分辨率
    保持长宽比，使用填充方式调整到目标尺寸，并保存原始尺寸信息
    """
    def __init__(self, target_size, interpolation=T.InterpolationMode.BILINEAR):
        self.interpolation = interpolation
        if isinstance(target_size, int):
            self.target_h, self.target_w = target_size, target_size
        else:
            self.target_h, self.target_w = target_size  # (H, W)

    def __call__(self, img_pil):
        original_w, original_h = img_pil.size  # PIL返回(width, height)
        original_size = (original_h, original_w)
        
        # 计算缩放比例，保持长宽比
        scale_h = self.target_h / original_h
        scale_w = self.target_w / original_w
        scale = min(scale_h, scale_w)  # 使用较小的缩放比例保持长宽比

        # 计算新的尺寸
        new_h = int(original_h * scale)
        new_w = int(original_w * scale)

        # 先缩放
        img_resized = TF.resize(img_pil, (new_h, new_w), interpolation=self.interpolation)

        # 再填充到目标尺寸
        pad_h = max(0, self.target_h - new_h)
        pad_w = max(0, self.target_w - new_w)

        # 计算填充量，使图像居中
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # 使用黑色填充
        img_padded = TF.pad(img_resized, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
        
        # 将原始尺寸信息附加到图像对象上（用于后续处理）
        img_padded.original_size = original_size
        
        return img_padded


class VisibilityDataset(Dataset):
    """
    高速公路能见度数据集处理
    """
    def __init__(self, image_paths, labels, is_train=True, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.is_train = is_train
        self.augment = augment
        self.target_size = config.TARGET_INPUT_SIZE
        self.base_transform = InputResize(self.target_size) # 尺寸变换
        self.normalize_transform = T.Compose([T.ToTensor(), T.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)]) # 标准化变换

        # 数据增强变换（默认不使用）
        if self.is_train and self.augment:
            # 保守的数据增强策略（可选）
            self.augment_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),  # 只保留最安全的水平翻转
                T.ColorJitter(brightness=0.05, contrast=0.05)  # 极轻微的颜色调整
            ])
        else:
            self.augment_transform = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(image_path).convert('RGB')
            
            # 统一使用基础变换（尺寸调整）
            image = self.base_transform(image)
            
            # 可选的数据增强（默认不使用）
            if self.augment_transform is not None:
                image = self.augment_transform(image)

            # 标准化
            image = self.normalize_transform(image)

            return image, label

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None


def collate_fn_filter_none(batch):
    """
    过滤掉None的批次数据
    """
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)
