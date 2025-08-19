import random
import numpy as np
from PIL import ImageEnhance, ImageFilter, ImageOps, Image, ImageDraw
import cv2


class VisAugmentation:    
    def __init__(self, enable_advanced_aug=True):
        # 定义不同能见度级别的增强强度
        self.augmentation_strength = {
            0: 'strong',      # 能见度好 - 强增强
            1: 'strong',      # 能见度好 - 强增强  
            2: 'medium',      # 中等能见度 - 中等增强
            3: 'strong',      # 能见度较差 - 强增强（关键优化点）
            4: 'light'        # 能见度最差 - 轻微增强
        }
        self.enable_advanced_aug = enable_advanced_aug
    
    def __call__(self, image, label):
        '''
        基于标签的智能数据增强
        
        Args:
            image: PIL Image对象
            label: 能见度标签 (0/1/2/3/4)
        '''
        
        strength = self.augmentation_strength[label]
        
        if strength == 'strong':
            image = self._apply_strong_augmentation(image, label)
        elif strength == 'medium':
            image = self._apply_medium_augmentation(image, label)
        elif strength == 'light':
            image = self._apply_light_augmentation(image, label)
        
        # 应用高级增强（如果启用）
        if self.enable_advanced_aug:
            image = self._apply_advanced_augmentation(image, label)
        
        return image
    
    def _apply_strong_augmentation(self, image, label):
        """强增强策略"""
        if label <= 1:
            # 0-1类（能见度好）：光照和色彩增强
            if random.random() < 0.6:
                factor = random.uniform(0.7, 1.3)
                image = ImageEnhance.Brightness(image).enhance(factor)
            
            if random.random() < 0.5:
                factor = random.uniform(0.9, 1.4)  # 适度增加对比度
                image = ImageEnhance.Contrast(image).enhance(factor)
            
            if random.random() < 0.5:
                factor = random.uniform(0.8, 1.2)
                image = ImageEnhance.Color(image).enhance(factor)
            
            # 添加锐化增强
            if random.random() < 0.3:
                image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
                
        elif label == 3:
            # 3类（能见度较差）：关键优化点 - 增强数据增强强度
            if random.random() < 0.7:  # 增加概率
                factor = random.uniform(0.8, 1.2)  # 扩大范围
                image = ImageEnhance.Brightness(image).enhance(factor)
            
            if random.random() < 0.6:  # 增加对比度调整
                factor = random.uniform(0.8, 1.3)
                image = ImageEnhance.Contrast(image).enhance(factor)
            
            # 添加锐化增强，帮助模型学习边缘特征
            if random.random() < 0.4:
                image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
            
            # 添加轻微噪声增强，模拟真实场景
            if random.random() < 0.3:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.3)))
            
            # 添加色彩增强，提高特征区分度
            if random.random() < 0.4:
                factor = random.uniform(0.9, 1.1)
                image = ImageEnhance.Color(image).enhance(factor)
        
        return image
    
    def _apply_medium_augmentation(self, image, label):
        """中等增强策略"""
        if label == 2:
            # 2类（中等能见度）：平衡增强
            if random.random() < 0.4:
                factor = random.uniform(0.85, 1.15)
                image = ImageEnhance.Brightness(image).enhance(factor)
            
            if random.random() < 0.3:
                factor = random.uniform(0.9, 1.1)
                image = ImageEnhance.Contrast(image).enhance(factor)
            
            # 添加轻微锐化
            if random.random() < 0.2:
                image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=2))
            
            # 轻微模糊
            if random.random() < 0.15:
                radius = random.uniform(0.1, 0.3)
                image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        return image
    
    def _apply_light_augmentation(self, image, label):
        """轻微增强策略"""
        if label == 4:
            # 4类（能见度最差）：保持原样，仅做极轻微调整
            if random.random() < 0.1:
                factor = random.uniform(0.98, 1.02)
                image = ImageEnhance.Brightness(image).enhance(factor)
        
        return image
    
    def _apply_advanced_augmentation(self, image, label):
        """高级增强策略"""
        # 几何变换
        if random.random() < 0.3:
            image = self._apply_geometric_augmentation(image, label)
        
        # 天气模拟
        if random.random() < 0.2:
            image = self._add_weather_simulation(image, label)
        
        # 噪声增强
        if random.random() < 0.15:
            image = self._add_noise_augmentation(image, label)
        
        return image
    
    def _apply_geometric_augmentation(self, image, label):
        """几何变换增强"""
        if label <= 2:  # 能见度较好的类别
            # 随机旋转（小角度）
            if random.random() < 0.3:
                angle = random.uniform(-5, 5)
                image = image.rotate(angle, fillcolor=(128, 128, 128))
            
            # 随机裁剪和填充
            if random.random() < 0.2:
                image = self._random_crop_and_pad(image, crop_ratio=0.9)
        
        elif label == 3:  # 第3类特殊处理
            # 轻微旋转，模拟拍摄角度变化
            if random.random() < 0.4:
                angle = random.uniform(-3, 3)
                image = image.rotate(angle, fillcolor=(128, 128, 128))
            
            # 随机裁剪，增加数据多样性
            if random.random() < 0.3:
                image = self._random_crop_and_pad(image, crop_ratio=0.95)
        
        return image
    
    def _random_crop_and_pad(self, image, crop_ratio):
        """随机裁剪和填充"""
        width, height = image.size
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)
        
        # 随机选择裁剪位置
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        
        # 裁剪
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        
        # 填充回原尺寸
        result = Image.new(image.mode, (width, height), (128, 128, 128))
        result.paste(cropped, ((width - crop_width) // 2, (height - crop_height) // 2))
        
        return result
    
    def _add_weather_simulation(self, image, label):
        """天气模拟增强"""
        if label >= 2:  # 针对能见度较差的类别
            if random.random() < 0.3:
                # 模拟雾霾效果
                fog_factor = random.uniform(0.1, 0.4)
                fog_color = (random.randint(180, 220), random.randint(180, 220), random.randint(180, 220))
                fog_layer = Image.new('RGB', image.size, fog_color)
                image = Image.blend(image, fog_layer, fog_factor)
            
            elif random.random() < 0.2:
                # 模拟雨滴效果
                image = self._add_rain_effect(image, intensity=random.uniform(0.1, 0.3))
        
        return image
    
    def _add_rain_effect(self, image, intensity):
        """添加雨滴效果"""
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 创建雨滴掩码
        height, width = img_array.shape[:2]
        rain_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 随机生成雨滴
        num_drops = int(width * height * intensity * 0.01)
        for _ in range(num_drops):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            length = random.randint(5, 15)
            
            # 绘制雨滴
            cv2.line(rain_mask, (x, y), (x, min(y + length, height - 1)), 255, 1)
        
        # 应用雨滴效果
        rain_effect = np.zeros_like(img_array)
        rain_effect[rain_mask > 0] = [200, 200, 200]  # 雨滴颜色
        
        # 混合原图和雨滴效果
        result = cv2.addWeighted(img_array, 1 - intensity, rain_effect, intensity, 0)
        
        return Image.fromarray(result.astype(np.uint8))
    
    def _add_noise_augmentation(self, image, label):
        """噪声增强"""
        if label >= 2:  # 能见度较差的类别
            if random.random() < 0.3:
                # 添加高斯噪声
                img_array = np.array(image)
                noise = np.random.normal(0, random.uniform(5, 15), img_array.shape)
                noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                image = Image.fromarray(noisy_img)
        
        return image