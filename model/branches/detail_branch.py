# model/branches/detail_branch.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetailBranch(nn.Module):
    '''
    使用导向滤波提取图像细节特征
    输入：RGB图像，形状为(B, 3, H, W)
    输出：细节层特征图，形状为(B, 3, H, W)
    '''
    def __init__(self, guided_radius, guided_eps):
        super().__init__()
        self.radius = guided_radius
        self.eps = guided_eps
        # 没有可训练参数

    def guided_filter(self, guide_img, input_img, radius=5, epsilon=1e-3):
        '''
        导向滤波，对transmission_map使用，可优化结果
        guide_img使用原图 (B, 3, H, W)
        input_img可以是灰度图或RGB图
        '''
        B, C, H, W = guide_img.shape
        
        # 如果input_img是灰度图，扩展为与导向图像相同的通道数
        if input_img.dim() == 3:
            input_img = input_img.unsqueeze(1)  # (B, 1, H, W)
        if input_img.shape[1] != C:
            input_img = input_img.expand(-1, C, -1, -1)  # (B, C, H, W)

        # 计算均值 使用平均池化实现box_filter的效果
        mean_guide = F.avg_pool2d(guide_img, kernel_size=2*radius+1, stride=1, padding=radius)  # (B, C, H, W)
        mean_input = F.avg_pool2d(input_img, kernel_size=2*radius+1, stride=1, padding=radius)  # (B, C, H, W)
        mean_guide_input = F.avg_pool2d(guide_img * input_img, kernel_size=2*radius+1, stride=1, padding=radius)  # (B, C, H, W)
        mean_guide_sq = F.avg_pool2d(guide_img * guide_img, kernel_size=2*radius+1, stride=1, padding=radius)  # (B, C, H, W)

        # 计算协方差和方差
        cov_guide_input = mean_guide_input - mean_guide * mean_input  # (B, C, H, W)
        var_guide = mean_guide_sq - mean_guide * mean_guide  # (B, C, H, W)

        # 计算线性系数
        a = cov_guide_input / (var_guide + epsilon)  # (B, C, H, W)
        b = mean_input - a * mean_guide  # (B, C, H, W)

        # 对系数进行均值滤波 对应计算像素所在区域的所有a,b的均值
        mean_a = F.avg_pool2d(a, kernel_size=2*radius+1, stride=1, padding=radius)  # (B, C, H, W)
        mean_b = F.avg_pool2d(b, kernel_size=2*radius+1, stride=1, padding=radius)  # (B, C, H, W)

        # 滤波结果，保持三通道
        output = mean_a * guide_img + mean_b  # (B, C, H, W)

        return output

    def denormalize(self, x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        '''
        反标准化
        '''
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(x.device)
        return x * std + mean

    def forward(self, x): # 输入是RGB Tensor (B, 3, H, W), 0-1或标准化后
        '''
        输入的x是经过ToTensor()和Normalize()后的，需要反标准化后再裁剪到0~1之间
        '''
        # 反标准化 + 裁剪到0~1之间
        x_denorm = self.denormalize(x)
        x_norm = torch.clamp(x_denorm, 0, 1)  # (B, 3, H, W)
        
        # 导向图：使用RGB提供丰富的边缘信息  
        # 输入图：使用灰度图，让RGB边缘信息指导灰度图的滤波
        gray_input = x_norm.mean(dim=1)  # (B, H, W) 灰度图
        base_layer = self.guided_filter(x_norm, gray_input, radius=self.radius, epsilon=self.eps)  # 返回 (B, 3, H, W)
        
        # 计算细节层: detail = original - base (保持三通道)
        detail_layer = x_norm - base_layer  # (B, 3, H, W)
        
        return detail_layer