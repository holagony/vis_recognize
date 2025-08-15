import random
from PIL import ImageEnhance, ImageFilter


class VisAugmentation:    
    def __call__(self, image, label):
        '''
        基于标签的数据增强
        
        Args:
            image: PIL Image对象
            label: 能见度标签 (0/1/2/3/4)
        '''
        
        if label <= 1:
            # 0-1类（能见度好）：光照和色彩增强，不降低能见度
            if random.random() < 0.5:
                factor = random.uniform(0.8, 1.2)
                image = ImageEnhance.Brightness(image).enhance(factor)
            
            if random.random() < 0.4:
                factor = random.uniform(1.0, 1.3)  # 只增加对比度
                image = ImageEnhance.Contrast(image).enhance(factor)
            
            if random.random() < 0.4:
                factor = random.uniform(0.8, 1.2)
                image = ImageEnhance.Color(image).enhance(factor)
                
        elif label == 2:
            # 2类（中等能见度）：轻度增强
            if random.random() < 0.3:
                factor = random.uniform(0.9, 1.1)
                image = ImageEnhance.Brightness(image).enhance(factor)
            
            if random.random() < 0.2:
                factor = random.uniform(0.9, 1.1)
                image = ImageEnhance.Contrast(image).enhance(factor)
            
            if random.random() < 0.1:
                radius = random.uniform(0.1, 0.4)
                image = image.filter(ImageFilter.GaussianBlur(radius=radius))
                
        elif label == 3:
            # 3类（能见度较差）：仅轻微亮度调整
            if random.random() < 0.2:
                factor = random.uniform(0.95, 1.05)
                image = ImageEnhance.Brightness(image).enhance(factor)
                
        else:
            # 4类（能见度最差）：几乎不变
            if random.random() < 0.1:
                factor = random.uniform(0.98, 1.02)
                image = ImageEnhance.Brightness(image).enhance(factor)
        
        return image