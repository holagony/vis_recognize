import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image
from utils import config
from datasets.vis_augmentation import VisAugmentation


class InputResize:
    '''
    自适应调整图像尺寸，支持任意输入尺寸，内部自动处理到指定分辨率
    
    支持两种模式：
    1. 保持长宽比 + 填充（默认）：图像保持原始长宽比，用黑色填充到目标尺寸
    2. 直接resize：直接缩放到目标尺寸，可能会改变长宽比
    
    参数:
        target_size: 目标尺寸，可以是int或tuple(H, W)
        interpolation: 插值方法
        direct_resize: 是否使用直接resize，False为保持长宽比模式（默认）
    '''
    def __init__(self, target_size, interpolation=T.InterpolationMode.BILINEAR, direct_resize=True):
        self.interpolation = interpolation
        self.direct_resize = direct_resize
        if isinstance(target_size, int):
            self.target_h, self.target_w = target_size, target_size
        else:
            self.target_h, self.target_w = target_size  # (H, W)

    def __call__(self, img_pil):
        original_w, original_h = img_pil.size  # PIL返回(width, height)
        original_size = (original_h, original_w)
        
        if self.direct_resize:
            img_resized = TF.resize(img_pil, (self.target_h, self.target_w), interpolation=self.interpolation)
            img_resized.original_size = original_size
            return img_resized
        else:
            # 保持长宽比模式：缩放 + 填充
            # 计算缩放比例，保持长宽比
            scale_h = self.target_h / original_h
            scale_w = self.target_w / original_w
            scale = min(scale_h, scale_w)  # 使用较小的缩放比例保持长宽比
    
            # 缩放 + 再填充到目标尺寸
            new_h = int(original_h * scale)
            new_w = int(original_w * scale)
            img_resized = TF.resize(img_pil, (new_h, new_w), interpolation=self.interpolation)
            pad_h = max(0, self.target_h - new_h)
            pad_w = max(0, self.target_w - new_w)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            img_padded = TF.pad(img_resized, [pad_left, pad_top, pad_right, pad_bottom], fill=0) # 黑色填充
            img_padded.original_size = original_size
            
            return img_padded


class VisibilityDataset(Dataset):
    '''
    高速公路能见度数据集处理，返回原始RGB图像和标签。
    
    image_paths: 图像路径列表
    labels: 标签列表
    augment: 是否使用数据增强
    is_train: 是否为训练模式（影响数据增强的应用）
    '''
    def __init__(self, image_paths, labels, augment, is_train=True):
        self.image_paths = image_paths
        self.labels = labels
        self.is_train = is_train
        self.augment = augment
        self.target_size = config.TARGET_INPUT_SIZE
        self.size_transform = InputResize(self.target_size, direct_resize=config.DIRECT_RESIZE)
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

        if self.is_train: # 概率水平翻转
            image = self.flip_transform(image)

        # 先转换为原始tensor，用于特征提取
        original_image = self.tensor_transform(image)
        
        # 创建增强后的tensor用于特征融合
        if self.is_train and self.augment:
            # 对PIL Image进行增强
            augmented_pil = self.augment_transform(image, label)
            augmented_image = self.tensor_transform(augmented_pil)
        else:
            # 验证/测试模式：不应用任何增强
            augmented_image = original_image.clone()

        # 返回原始图像和增强图像，不进行特征提取
        # original_image用于物理特征提取，augmented_image用于特征融合
        return original_image, augmented_image, label
