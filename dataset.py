import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import torchvision.transforms as T
from PIL import Image
from utils.config import TARGET_INPUT_SIZE, NORM_MEAN, NORM_STD


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


class VisAugmentation:
    """
    针对能见度识别的数据增强
    """
    def __init__(self, label):
        self.label = label  # 能见度等级标签
    
    def __call__(self, img):
        """根据能见度等级进行智能增强"""
        # 转换为numpy数组便于处理
        img_array = np.array(img)
        
        # 根据能见度等级调整增强策略
        if self.label == 0:  # 极低能见度
            img_array = self._simulate_low_visibility(img_array)
        elif self.label == 1:  # 低能见度
            img_array = self._simulate_medium_low_visibility(img_array)
        elif self.label == 2:  # 中等能见度
            img_array = self._simulate_medium_visibility(img_array)
        elif self.label == 3:  # 高能见度
            img_array = self._simulate_high_visibility(img_array)
        # label == 4 (极高能见度) 保持原样
        
        return Image.fromarray(img_array)
    
    def _simulate_low_visibility(self, img):
        """模拟极低能见度条件"""
        # 大幅降低对比度和亮度
        img = img * 0.3  # 降低亮度
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # 添加雾霾效果
        if random.random() < 0.7:
            img = self._add_fog_effect(img, intensity=0.8)
        
        return img
    
    def _simulate_medium_low_visibility(self, img):
        """模拟低能见度条件"""
        # 适度降低对比度和亮度
        img = img * 0.6
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # 轻微雾霾效果
        if random.random() < 0.5:
            img = self._add_fog_effect(img, intensity=0.4)
        
        return img
    
    def _simulate_medium_visibility(self, img):
        """模拟中等能见度条件"""
        # 轻微降低对比度
        img = img * 0.8
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # 轻微雾霾效果
        if random.random() < 0.3:
            img = self._add_fog_effect(img, intensity=0.2)
        
        return img
    
    def _simulate_high_visibility(self, img):
        """模拟高能见度条件"""
        # 轻微提升对比度
        img = img * 1.1
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img
    
    def _add_fog_effect(self, img, intensity=0.5):
        """添加雾霾效果"""
        h, w = img.shape[:2]
        
        # 创建雾霾层
        fog = np.ones((h, w, 3), dtype=np.uint8) * 200  # 白色雾霾
        
        # 根据距离中心点的距离调整雾霾强度
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 距离越远，雾霾越浓
        fog_factor = distance / max_distance * intensity
        fog_factor = np.clip(fog_factor, 0, 1)
        
        # 应用雾霾效果
        fog_factor = np.stack([fog_factor] * 3, axis=2)
        img = img * (1 - fog_factor) + fog * fog_factor
        
        return img.astype(np.uint8)


class EnvironmentalNoiseTransform:
    """
    可序列化的环境噪声变换类，替代Lambda函数
    """
    def __init__(self, noise_prob=0.2, noise_std=0.01):
        self.noise_prob = noise_prob
        self.noise_std = noise_std
    
    def __call__(self, img):
        """
        添加环境噪声，模拟真实摄像头环境
        """
        if random.random() < self.noise_prob:  # 默认20%概率添加噪声
            img_array = TF.to_tensor(img)
            noise = torch.randn_like(img_array) * self.noise_std
            img_array = torch.clamp(img_array + noise, 0, 1)
            img = TF.to_pil_image(img_array)
        return img


class VisibilityDataset(Dataset):
    """
    高速公路能见度数据集处理
    """
    def __init__(self, image_paths, labels, is_train=True, augment=True):
        self.image_paths = image_paths
        self.labels = labels
        self.is_train = is_train
        self.augment = augment
        self.target_size = TARGET_INPUT_SIZE
        self.base_transform = InputResize(self.target_size) # 尺寸变换
        self.normalize_transform = T.Compose([T.ToTensor(), T.Normalize(mean=NORM_MEAN, std=NORM_STD)]) # 标准化变换

        # 数据增强变换
        if self.is_train and self.augment:
            self.augment_transform = T.Compose([
                T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.02, hue=0.01), # 轻微的颜色调整，避免过度影响能见度判断
                T.RandomHorizontalFlip(p=0.3), # 水平翻转（高速公路场景通常可以翻转）
                T.RandomRotation(degrees=2, fill=128),  # 使用灰色填充，角度更小
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)), # 轻微模糊，模拟不同能见度条件
                EnvironmentalNoiseTransform()]) # 添加轻微噪声，模拟真实环境
        else:
            self.augment_transform = None



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(image_path).convert('RGB')
            image = self.base_transform(image) # 尺寸调整
            original_size = None
            if hasattr(image, 'original_size'):
                original_size = image.original_size

            if self.augment_transform is not None: # 数据增强（仅训练时）
                image = self.augment_transform(image)
                # visibility_aug = VisAugmentation(label) # 根据能见度等级进行智能增强
                # image = visibility_aug(image)

            # 标准化
            image = self.normalize_transform(image)
            
            # 如果有原始尺寸信息，将其附加到tensor上
            if original_size is not None:
                image.original_size = original_size

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
