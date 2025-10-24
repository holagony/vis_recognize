import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image
from utils import config
from datasets.vis_augmentation import VisAugmentation


class InputResize:
    '''    
    1. 直接resize：直接缩放到目标尺寸（改变长宽比）
    2. 保持长宽比 + 填充：图像保持原始长宽比，用黑色填充到目标尺寸
    3. 中心裁剪：先缩放到合适尺寸，然后从中心裁剪到目标尺寸
    4. 随机裁剪：先缩放到合适尺寸，然后随机裁剪到目标尺寸（训练时推荐）
    5. 随机缩放裁剪：随机缩放和裁剪，提供更强的数据增强效果
    target_size: 目标尺寸，可以是int或tuple(H, W)
    resize_mode: 'direct', 'pad', 'center_crop', 'random_crop', 'random_resized_crop'
    '''

    def __init__(self, target_size, is_train=True, resize_mode='center_crop'):
        self.interpolation = T.InterpolationMode.BILINEAR
        self.is_train = is_train
        self.resize_mode = resize_mode

        if isinstance(target_size, int):
            self.target_h, self.target_w = target_size, target_size
        else:
            self.target_h, self.target_w = target_size  # (H, W)

    def _calculate_padding(self, new_h, new_w):
        pad_h = max(0, self.target_h - new_h)
        pad_w = max(0, self.target_w - new_w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return (pad_left, pad_top, pad_right, pad_bottom)

    def _resize_and_center_crop(self, img_pil, original_h, original_w):
        scale_h = self.target_h / original_h
        scale_w = self.target_w / original_w
        scale = max(scale_h, scale_w)
        new_h = int(original_h * scale)
        new_w = int(original_w * scale)
        transform = T.Compose([T.Resize(size=(new_h, new_w), interpolation=self.interpolation), T.CenterCrop(size=(self.target_h, self.target_w))])
        return transform(img_pil)

    def __call__(self, img_pil):
        original_w, original_h = img_pil.size  # PIL返回(width, height)
        original_size = (original_h, original_w)

        if self.resize_mode == 'direct':  # 直接resize
            transform = T.Resize(size=(self.target_h, self.target_w), interpolation=self.interpolation)
            img_resized = transform(img_pil)
            img_resized.original_size = original_size
            return img_resized

        elif self.resize_mode == 'pad':  # resize保持长宽比 + 填充黑色
            scale_h = self.target_h / original_h
            scale_w = self.target_w / original_w
            scale = min(scale_h, scale_w)
            new_h = int(original_h * scale)
            new_w = int(original_w * scale)
            transform = T.Compose(
                [T.Resize(size=(new_h, new_w), interpolation=self.interpolation),
                 T.Pad(padding=self._calculate_padding(new_h, new_w), fill=0)])
            img_padded = transform(img_pil)
            img_padded.original_size = original_size
            return img_padded

        elif self.resize_mode == 'center_crop':  # 中心裁剪
            img_cropped = self._resize_and_center_crop(img_pil, original_h, original_w)
            img_cropped.original_size = original_size
            return img_cropped

        elif self.resize_mode == 'random_crop':  # 随机裁剪 Resize + RandomCrop组合
            scale_h = self.target_h / original_h
            scale_w = self.target_w / original_w
            scale = max(scale_h, scale_w)
            new_h = int(original_h * scale)
            new_w = int(original_w * scale)

            if self.is_train:
                transform = T.Compose(
                    [T.Resize(size=(new_h, new_w), interpolation=self.interpolation),
                     T.RandomCrop(size=(self.target_h, self.target_w))])
                img_cropped = transform(img_pil)
            else:
                img_cropped = self._resize_and_center_crop(img_pil, original_h, original_w)

            img_cropped.original_size = original_size
            return img_cropped

        elif self.resize_mode == 'random_resized_crop':  # 随机缩放裁剪
            if self.is_train:
                transform = T.RandomResizedCrop(size=(self.target_h, self.target_w),
                                                scale=(0.6, 1.0),
                                                ratio=(0.75, 1.33),
                                                interpolation=self.interpolation)
                img_cropped = transform(img_pil)
            else:
                img_cropped = self._resize_and_center_crop(img_pil, original_h, original_w)

            img_cropped.original_size = original_size
            return img_cropped


class VisibilityDataset(Dataset):
    '''
    高速公路能见度数据集构建
    image_paths: 图像路径列表
    labels: 标签列表
    augment: 是否使用数据增强
    is_train: 是否为训练模式（影响数据增强的应用）
    resize_mode: 图像处理模式 ('direct', 'pad', 'center_crop', 'random_crop', 'random_resized_crop')
    '''

    def __init__(self, image_paths, labels, augment, is_train=True, resize_mode='center_crop'):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment
        self.is_train = is_train
        self.resize_mode = resize_mode
        self.target_size = config.TARGET_INPUT_SIZE
        self.size_transform = InputResize(self.target_size, resize_mode=self.resize_mode, is_train=self.is_train)
        self.flip_transform = T.RandomHorizontalFlip(p=0.5)
        self.tensor_transform = T.ToTensor()

        if self.is_train and self.augment:
            self.augment_transform = VisAugmentation()
        else:
            self.augment_transform = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        label = int(label)

        image = Image.open(image_path).convert('RGB')
        image = self.size_transform(image)

        if self.is_train:  # 概率水平翻转
            image = self.flip_transform(image)

        # 先转换为原始tensor，用于特征提取
        original_image = self.tensor_transform(image)

        # 创建增强后的tensor用于特征融合
        if self.is_train and self.augment:  # 启动数据增强
            augmented_pil = self.augment_transform(image, label)
            augmented_image = self.tensor_transform(augmented_pil)
        else:
            augmented_image = original_image.clone()  # 不增强

        # 返回原始图像和增强图像，不进行特征提取
        # 目前没有数据增强，两个tensor是一样的
        # original_image用于物理特征提取，augmented_image用于特征融合
        return original_image, augmented_image, label
