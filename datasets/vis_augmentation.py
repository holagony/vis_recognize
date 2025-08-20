import random
import math
import cv2
import numpy as np
from PIL import ImageEnhance, ImageFilter, Image


class VisAugmentation:

    def __init__(self):
        # 定义不同能见度级别的增强策略
        self.augmentation_strategy = {0: 'none', 1: 'none', 2: 'medium', 3: 'strong', 4: 'light'}

        # 大气散射模型参数
        self.fog_params = {
            'beta_range': (0.02, 0.15),  # 散射系数范围，影响雾霾密度
            'A_range': (180, 240),  # 大气光强度范围，影响雾霾亮度
            'd_range': (0.2, 1.0)  # 深度范围，影响雾霾的空间分布
        }

    def __call__(self, image, label):
        '''
        基于标签的智能数据增强
        '''
        strategy = self.augmentation_strategy[label]

        if strategy == 'none':
            return image
        elif strategy == 'medium': # 中等增强
            image = self._apply_medium_augmentation(image, label)
        elif strategy == 'strong': # 强增强
            image = self._apply_strong_augmentation(image, label)
        elif strategy == 'light': # 轻微增强
            image = self._apply_light_augmentation(image, label)

        return image

    def _apply_medium_augmentation(self, image, label):
        """中等增强策略 - 2类（中等能见度）"""
        # 中等强度的物理模型增强
        if random.random() < 0.6:  # 中等概率应用
            fog_intensity = self._get_fog_intensity_by_label(label)
            image = self._add_atmospheric_scattering_fog(image, fog_intensity)

        # 中等模糊
        if random.random() < 0.4:
            blur_radius = self._get_blur_radius_by_label(label)
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # 中等对比度调整
        if random.random() < 0.3:
            contrast_factor = self._get_contrast_factor_by_label(label)
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        return image

    def _apply_strong_augmentation(self, image, label):
        """强增强策略 - 3类（能见度较差）"""
        # 高强度的物理模型增强
        if random.random() < 0.9:  # 高概率应用
            fog_intensity = self._get_fog_intensity_by_label(label)
            image = self._add_atmospheric_scattering_fog(image, fog_intensity)

        # 强模糊
        if random.random() < 0.7:
            blur_radius = self._get_blur_radius_by_label(label)
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # 强对比度调整
        if random.random() < 0.6:
            contrast_factor = self._get_contrast_factor_by_label(label)
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        # 额外的噪声增强
        if random.random() < 0.3:
            image = self._add_noise_augmentation(image, label)

        return image

    def _apply_light_augmentation(self, image, label):
        """轻微增强策略 - 4类（能见度最差）"""
        # 轻微强度的物理模型增强
        if random.random() < 0.4:  # 低概率应用
            fog_intensity = self._get_fog_intensity_by_label(label)
            image = self._add_atmospheric_scattering_fog(image, fog_intensity)

        # 轻微模糊
        if random.random() < 0.2:
            blur_radius = self._get_blur_radius_by_label(label)
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # 轻微对比度调整
        if random.random() < 0.15:
            contrast_factor = self._get_contrast_factor_by_label(label)
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        return image

    def _get_fog_intensity_by_label(self, label):
        """根据标签确定雾霾强度 - 确保物理合理性"""
        if label == 0:  # 能见度好
            return random.uniform(0.1, 0.3)  # 轻微雾霾
        elif label == 3:  # 能见度较差
            return random.uniform(0.4, 0.7)  # 中等雾霾
        elif label == 4:  # 能见度最差
            return random.uniform(0.6, 0.9)  # 重度雾霾
        else:
            return random.uniform(0.2, 0.5)  # 默认中等强度

    def _get_blur_radius_by_label(self, label):
        """根据标签确定模糊半径 - 确保物理合理性"""
        if label == 0:  # 能见度好
            return random.uniform(0.1, 0.3)  # 轻微模糊
        elif label == 3:  # 能见度较差
            return random.uniform(0.3, 0.8)  # 中等模糊
        elif label == 4:  # 能见度最差
            return random.uniform(0.5, 1.2)  # 重度模糊
        else:
            return random.uniform(0.2, 0.6)  # 默认中等强度

    def _get_contrast_factor_by_label(self, label):
        """根据标签确定对比度因子 - 确保物理合理性"""
        if label == 0:  # 能见度好
            return random.uniform(0.85, 1.1)  # 轻微调整
        elif label == 3:  # 能见度较差
            return random.uniform(0.6, 0.85)  # 降低对比度
        elif label == 4:  # 能见度最差
            return random.uniform(0.5, 0.75)  # 显著降低对比度
        else:
            return random.uniform(0.7, 0.9)  # 默认中等调整

    def _add_atmospheric_scattering_fog(self, image, intensity):
        """大气散射模型 - 基于物理规律的雾霾合成"""
        img_array = np.array(image).astype(np.float32) / 255.0
        height, width = img_array.shape[:2]

        # 根据强度调整物理参数
        beta = intensity * random.uniform(0.8, 1.2)  # 散射系数与强度成正比
        A = random.uniform(*self.fog_params['A_range']) / 255.0

        # 创建更真实的深度图 - 使用图像梯度信息
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # 基于梯度的深度估计：边缘区域深度较小，平坦区域深度较大
        depth_map = 0.1 + 0.7 * (1.0 - gradient_magnitude / np.max(gradient_magnitude))
        depth_map = np.clip(depth_map, 0.1, 0.8)

        # 计算透射率：t(x) = exp(-β * d(x)) - 物理散射模型
        transmission = np.exp(-beta * depth_map)

        # 应用大气散射模型：I(x) = J(x)t(x) + A(1-t(x))
        # 其中：J(x)是清晰图像，t(x)是透射率，A是大气光
        foggy_image = img_array * transmission[:, :, np.newaxis] + A * (1 - transmission[:, :, np.newaxis])

        # 确保像素值在合理范围内
        foggy_image = np.clip(foggy_image, 0, 1)

        # 转换回uint8
        foggy_image = (foggy_image * 255).astype(np.uint8)

        return Image.fromarray(foggy_image)

    def _add_noise_augmentation(self, image, label):
        """传感器噪声增强 - 仅在必要时添加"""
        if label >= 3 and random.random() < 0.2:  # 仅对低能见度类别，低概率
            img_array = np.array(image)
            # 添加轻微的高斯噪声，模拟传感器噪声
            noise_std = random.uniform(2, 8)  # 较小的噪声标准差
            noise = np.random.normal(0, noise_std, img_array.shape)
            noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(noisy_img)

        return image
