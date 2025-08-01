# model/branches/spectral_branch.py
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class SpectralBranch(nn.Module):
    def __init__(self, enhancement_factor=2.0): # 来自你的配置
        super().__init__()
        self.enhancement_factor = enhancement_factor

    def rgb_to_lab(self, image):
        """
        将RGB图像转换为简化的LAB色彩空间表示
        输入RGB图像，形状为 (B, C, H, W)，C=3，值范围[0,1]
        输出LAB图像，形状为 (B, C, H, W)，C=3
        """
        # 简化的RGB到LAB转换：
        # L通道：亮度（标准灰度计算）范围[0,1]
        # A通道：红绿差异，范围[-0.5,0.5]  
        # B通道：蓝黄差异，范围[-0.5,0.5]
        R = image[:, 0:1, :, :]
        G = image[:, 1:2, :, :]
        B = image[:, 2:3, :, :]
        
        # L通道：标准亮度计算
        L = 0.299 * R + 0.587 * G + 0.114 * B  # 范围[0,1]
        
        # A通道：红绿对比 
        A = (R - G) * 0.5  # 范围[-0.5,0.5]
        
        # B通道：蓝黄对比，修正计算
        yellow = (R + G) * 0.5  # 黄色分量
        B_ch = (B - yellow) * 0.5  # 蓝-黄差异，范围[-0.5,0.5]
        
        lab_image = torch.cat([L, A, B_ch], dim=1)
        
        return lab_image

    def normalize_lab(self, lab_image):
        '''
        对我们自定义的LAB图像进行标准化处理
        实际范围：L[0,1], A[-0.5,0.5], B[-0.5,0.5]
        目标：统一标准化到合理范围用于增强
        '''
        lab_normalized = lab_image.clone()
        # L通道已经在[0,1]范围，保持不变
        lab_normalized[:, 0, :, :] = lab_image[:, 0, :, :]  # L 通道保持 [0, 1]
        # A通道从[-0.5,0.5]标准化到[-1,1]
        lab_normalized[:, 1, :, :] = lab_image[:, 1, :, :] * 2.0  # A 通道 [-0.5,0.5] -> [-1,1]
        # B通道从[-0.5,0.5]标准化到[-1,1] 
        lab_normalized[:, 2, :, :] = lab_image[:, 2, :, :] * 2.0  # B 通道 [-0.5,0.5] -> [-1,1]
        
        return lab_normalized

    def apply_spectral_enhancement(self, lab_normalized):
        '''
        对LAB图像的A和B通道进行增强。
        '''
        # 分离 L, A, B 通道
        L = lab_normalized[:, 0:1, :, :]  # (B, 1, H, W)
        A = lab_normalized[:, 1:2, :, :]  # (B, 1, H, W)
        B = lab_normalized[:, 2:3, :, :]  # (B, 1, H, W)
        
        # 对 A 和 B 通道进行增强
        A_enhanced = A * self.enhancement_factor
        B_enhanced = B * self.enhancement_factor
        
        # 裁剪回 [-1, 1] 范围
        A_enhanced = torch.clamp(A_enhanced, -1.0, 1.0)
        B_enhanced = torch.clamp(B_enhanced, -1.0, 1.0)
        
        # 重新组合
        enhanced_lab = torch.cat([L, A_enhanced, B_enhanced], dim=1)
        
        return enhanced_lab
    

    def denormalize(self, x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        '''
        反标准化
        '''
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(x.device)
        return x * std + mean


    def forward(self, x):
        """
        输入的x是经过ToTensor()和Normalize()后的，需要反标准化后再裁剪到0~1之间
        输出: 处理后的LAB特征图 (B, 3, H, W)
        """
        # 反标准化 + 裁剪到0~1之间
        x_denorm = self.denormalize(x)
        x_norm = torch.clamp(x_denorm, 0, 1)  # (B, 3, H, W)
        
        lab_image = self.rgb_to_lab(x_norm)
        lab_normalized = self.normalize_lab(lab_image)
        enhanced_lab = self.apply_spectral_enhancement(lab_normalized)
        
        return enhanced_lab