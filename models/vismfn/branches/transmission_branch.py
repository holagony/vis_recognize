# model/branches/transmission_branch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransmissionBranch(nn.Module):
    def __init__(self, omega, patch_size, guided_radius, guided_eps):
        super().__init__()
        self.omega = omega
        self.patch_size = patch_size
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps
        # 这个分支没有可训练参数，所有操作都是图像处理

    def denormalize(self, x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        '''
        反标准化
        '''
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(x.device)
        return x * std + mean

    def calculate_dark_channel(self, image, patch_size=5):
        """
        计算暗通道图像

        参数:
        - image: 输入图像 (B, C, H, W)
        - patch_size: 局部窗口大小，设置为5

        返回:
        - dark_channel: 暗通道图像 (B, H, W)
        """
        min_channel = torch.min(image, dim=1)[0]  # (B, H, W)

        # 使用最大池化操作计算局部最小值 min(x) = -max(-x)
        padding = patch_size // 2 # padding = (patch_size - 1) / 2
        dark_channel = -F.max_pool2d(-min_channel.unsqueeze(1), kernel_size=patch_size, stride=1, padding=padding).squeeze(1)  # (B, H, W)

        return dark_channel

    def estimate_atmosphere_light(self, image, dark_channel, top_k=0.001):
        """
        估计全局大气光值（批量处理版本）

        参数:
        - image: 输入图像 (B, C, H, W)
        - dark_channel: 暗通道图像 (B, H, W)
        - top_k: 选择最亮像素的比例，默认值为0.001 前0.1%

        返回:
        - atmosphere_light: 全局大气光值 (B, C)
        """
        B, C, H, W = image.shape

        # 选择暗通道中最亮的像素
        num_pixels = max(1, int(H * W * top_k))  # 至少选择1个像素
        flat_dark = dark_channel.view(B, -1)  # (B, H*W)
        _, indices = torch.topk(flat_dark, k=num_pixels, dim=1)  # (B, num_pixels)

        # 批量获取对应像素值
        flat_image = image.view(B, C, -1)  # (B, C, H*W)
        
        # 使用gather进行批量索引，避免循环
        # indices需要扩展到匹配flat_image的维度
        indices_expanded = indices.unsqueeze(1).expand(-1, C, -1)  # (B, C, num_pixels)
        selected_pixels = torch.gather(flat_image, 2, indices_expanded)  # (B, C, num_pixels)
        
        # 对每个批次和通道取最大值
        atmosphere_light = torch.max(selected_pixels, dim=2)[0]  # (B, C)

        return atmosphere_light

    def calculate_transmission_map(self, dark_channel, atmosphere_light, omega=0.95):
        """
        计算传输矩阵 t
        传输矩阵是大气消光系数与场景深度乘积的负指数函数，能反映能见度

        参数:
        - dark_channel: 暗通道图像 (B, H, W)
        - atmosphere_light: 全局大气光值 (B, C)
        - omega: 保留雾的程度，默认值为0.95

        返回:
        - transmission_map: 传输矩阵 t (B, H, W)
        """
        B, C = atmosphere_light.shape
        _, H, W = dark_channel.shape

        # 将大气光值扩展为与暗通道图像相同的形状
        atmosphere_light = atmosphere_light.view(B, C, 1, 1).expand(-1, -1, H, W)  # (B, C, H, W)

        # 计算t
        transmission_map = 1 - omega * (dark_channel.unsqueeze(1) / atmosphere_light.mean(dim=1, keepdim=True))  # (B, 1, H, W)
        transmission_map = torch.clamp(transmission_map, 0, 1) # 限制数值范围 0~1

        return transmission_map

    def guided_filter(self, guide_img, input_img, radius=5, epsilon=1e-3):
        """
        导向滤波，对transmission_map使用，优化结果
        guide_img使用原图
        input_img使用transmission_map

        参数:
        - guide: 导向图像 (B, C, H, W)
        - input_img: 输入图像 (B, H, W)
        - radius: 滤波窗口半径
        - epsilon: 正则化参数

        返回:
        - output: 滤波后的图像 (B, H, W)
        """
        B, C, H, W = guide_img.shape

        # 确保input_img有正确的维度
        if input_img.dim() == 3:  # (B, H, W)
            input_img = input_img.unsqueeze(1)  # (B, 1, H, W)
        
        # 将输入扩展为与导向图像相同的通道数
        input_img = input_img.expand(B, C, H, W)  # (B, C, H, W)

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

        # 滤波结果
        output = mean_a * guide_img + mean_b  # (B, C, H, W)
        # 由于input_img是单通道扩展的，所以3个通道结果相同，直接取第一个通道
        output = output[:, 0, :, :]  # (B, H, W)

        return output

    def forward(self, x):
        """
        输入 x_rgb_tensor_batch: PyTorch Tensor (B, 3, H, W)，经过ToTensor()和Normalize()后的
        输出: 传输矩阵 (B, 1, H, W)
        """
        # 反标准化 + 裁剪到0~1之间
        x_denorm = self.denormalize(x)
        x_norm = torch.clamp(x_denorm, 0, 1)  # (B, 3, H, W)

        # 批量计算暗通道图像
        dark_channel = self.calculate_dark_channel(x_norm, self.patch_size)  # (B, H, W)

        # 批量估计全局大气光值
        atmosphere_light = self.estimate_atmosphere_light(x_norm, dark_channel)  # (B, 3)

        # 批量计算传输矩阵
        transmission_map = self.calculate_transmission_map(dark_channel, atmosphere_light, self.omega)  # (B, 1, H, W)

        # 批量导向滤波优化传输矩阵
        refined_transmission_map = self.guided_filter(
            x_norm,                          # 导向图：RGB图像 (B, 3, H, W)
            transmission_map.squeeze(1),     # 输入图：传输图 (B, H, W)
            radius=self.guided_radius, 
            epsilon=self.guided_eps)  # (B, H, W)
        
        return refined_transmission_map.unsqueeze(1)  # (B, 1, H, W)
