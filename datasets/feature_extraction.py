import torch
import torch.nn as nn
import torch.nn.functional as F
from model_hub.dptransformer.dpt.models import DPTDepthModel
from torchvision.models import mobilenet_v2
from torchvision.transforms import functional as TF
from utils import config

'''
四个分支：
1. 深度估计分支，输出(1 or 16, H, W) - 1/16通道深度图，数值范围[-1, 1]
2. 细节特征提取分支，输出(3, H, W) - 三通道细节特征，数值范围[-1, 1]
3. 光谱特征提取分支，输出(3, H, W) - 三通道LAB特征，L[0,1], A[-0.5,0.5], B[-0.5,0.5]
4. 传输矩阵估计分支，输出(1, H, W) - 单通道传输矩阵，数值范围[0, 1]

RGB: [0, 1] - 标准图像值范围
深度: [-1, 1] - 相对深度信息，正负表示前后关系
传输: [0, 1] - 能见度信息，0表示完全不可见，1表示完全可见
光谱: L[0,1], A[-0.5,0.5], B[-0.5,0.5] - 色彩空间标准范围
细节: [-1, 1] - 细节增强，正负表示增强和抑制
'''

#------------------------------深度估计分支--------------------------------
class DPTSceneDepthBranch(nn.Module):
    '''
    DPT模型内部固定使用384x384，但外部可以接受任意尺寸，会自动调整到384x384
    DPT模型可选"dpt_large" or "dpt_hybrid"
    '''

    def __init__(self, dpt_model_type='dpt_hybrid', device='cpu'):
        super().__init__()
        self.device = device

        if dpt_model_type == "dpt_large":
            dpt_weight_path = "dpt/weights/dpt_large-midas-2f21e586.pt"
            backbone = "vitl16_384"

        elif dpt_model_type == "dpt_hybrid":
            dpt_weight_path = "dpt/weights/dpt_hybrid-midas-501f0c75.pt"
            backbone = "vitb_rn50_384"

        # 创建DPT模型
        self.dpt_model = DPTDepthModel(path=dpt_weight_path, backbone=backbone, non_negative=True, enable_attention_hooks=False)
        for param in self.dpt_model.parameters():  # 冻结参数
            param.requires_grad = False

        # 将整个模块移动到指定设备
        self.to(self.device)
        self.dpt_model.eval()

    def forward(self, x):
        original_size = x.shape[2:]
        with torch.no_grad():
            depth_map = self.dpt_model(x)

        # 将深度图调整回原始输入尺寸
        if depth_map.shape[2:] != original_size:
            depth_map = F.interpolate(depth_map.unsqueeze(1), size=original_size, mode='bicubic', align_corners=False)
        else:
            depth_map = depth_map.unsqueeze(1)

        return depth_map


class MobileNetEncoder(nn.Module):

    def __init__(self, device='cpu'):  # 移除 model_weight_path 参数
        super().__init__()
        self.device = device  # 保存 device

        # 创建 MobileNetV2 结构（不加载权重，权重将在外层统一加载）
        mobilenet_model_struct = mobilenet_v2(weights=None)

        self.features = mobilenet_model_struct.features
        self.output_channels = 1280
        self.layer0 = self.features[0:2]
        self.layer1 = self.features[2:4]
        self.layer2 = self.features[4:7]
        self.layer3 = self.features[7:14]
        self.layer4 = self.features[14:18]
        self.to(self.device)

    def forward(self, x):
        # 确保输入在同一设备
        # x = x.to(self.device)
        skips = {}
        x0 = self.layer0(x)
        skips['skip1'] = x0
        x1 = self.layer1(x0)
        skips['skip2'] = x1
        x2 = self.layer2(x1)
        skips['skip3'] = x2
        x3 = self.layer3(x2)
        skips['skip4'] = x3
        bottleneck = self.layer4(x3)
        return bottleneck, skips


class MobileNetDecoder(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=5):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x, skip_feature):
        x = self.upsample(x)
        if x.shape[2:] != skip_feature.shape[2:]:
            x = TF.resize(x, skip_feature.shape[2:])
        x = torch.cat([x, skip_feature], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SimpleSceneDepthBranch(nn.Module):

    def __init__(self, model_weight_path, freeze_weights=True, device='cpu'):
        super().__init__()
        self.device = device  # 保存 device
        self.encoder = MobileNetEncoder(device=self.device)

        bottleneck_channels = 320
        skip_channels_map = {'skip4': 96, 'skip3': 32, 'skip2': 24, 'skip1': 16}

        self.dec4 = MobileNetDecoder(bottleneck_channels, skip_channels_map['skip4'], 128)
        self.dec3 = MobileNetDecoder(128, skip_channels_map['skip3'], 64)
        self.dec2 = MobileNetDecoder(64, skip_channels_map['skip2'], 32)
        self.dec1 = MobileNetDecoder(32, skip_channels_map['skip1'], 16)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

        if model_weight_path:
            checkpoint = torch.load(model_weight_path, map_location=self.device, weights_only=False)

            # 处理不同的权重文件格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # 移除 'model.' 前缀（如果存在）
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        new_state_dict[k[len("model."):]] = v
                    else:
                        new_state_dict[k] = v
                state_dict = new_state_dict
            else:
                # 直接是state_dict格式
                state_dict = checkpoint

            try:
                self.load_state_dict(state_dict, strict=True)
            except:
                self.load_state_dict(state_dict, strict=False)

        if freeze_weights:
            for param in self.parameters():
                param.requires_grad = False

        self.to(self.device)

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        d4 = self.dec4(bottleneck, skips['skip4'])
        d3 = self.dec3(d4, skips['skip3'])
        d2 = self.dec2(d3, skips['skip2'])
        d1 = self.dec1(d2, skips['skip1'])
        features_before_final_conv = self.final_upsample(d1)
        output = self.final_conv(features_before_final_conv)

        # 添加激活函数限制输出范围，避免数值过大
        output = torch.tanh(output)  # 限制到[-1, 1]

        return output


#------------------------------细节特征提取分支--------------------------------
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

    def guided_filter(self, guide_img, input_img, radius=5, epsilon=1e-3):
        '''
        导向滤波，对transmission_map使用，可优化结果
        guide_img使用原图(B, 3, H, W)
        input_img可以是灰度图或RGB图(B, 1, H, W) 或 (B, 3, H, W)
        '''
        B, C, H, W = guide_img.shape


        input_img = input_img.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        input_img = input_img.expand(B, C, H, W)  # (B, 1, H, W) -> (B, C, H, W)

        # 计算均值 使用平均池化实现box_filter的效果
        mean_guide = F.avg_pool2d(guide_img, kernel_size=2 * radius + 1, stride=1, padding=radius)  # (B, 3, H, W)
        mean_input = F.avg_pool2d(input_img, kernel_size=2 * radius + 1, stride=1, padding=radius)  # (B, C, H, W)
        mean_guide_input = F.avg_pool2d((guide_img * input_img), kernel_size=2 * radius + 1, stride=1, padding=radius)  # (B, C, H, W)
        mean_guide_sq = F.avg_pool2d((guide_img * guide_img), kernel_size=2 * radius + 1, stride=1, padding=radius)  # (B, 3, H, W)

        # 计算协方差和方差
        cov_guide_input = mean_guide_input - mean_guide * mean_input  # (B, C, H, W)
        var_guide = mean_guide_sq - mean_guide * mean_guide  # (B, 3, H, W)

        # 计算线性系数
        a = cov_guide_input / (var_guide + epsilon)  # (B, C, H, W)
        b = mean_input - a * mean_guide  # (B, C, H, W)

        # 对系数进行均值滤波 对应计算像素所在区域的所有a,b的均值
        mean_a = F.avg_pool2d(a, kernel_size=2 * radius + 1, stride=1, padding=radius)  # (B, C, H, W)
        mean_b = F.avg_pool2d(b, kernel_size=2 * radius + 1, stride=1, padding=radius)  # (B, C, H, W)

        # 滤波结果，保持三通道
        output = mean_a * guide_img + mean_b  # (B, C, H, W)
        
        return output

    def forward(self, x):  # 输入是RGB Tensor (B, 3, H, W), 0-1范围
        '''
        输入的x是经过ToTensor()后的，值范围[0,1]
        '''
        # 导向图：使用RGB提供丰富的边缘信息
        # 输入图：使用灰度图，让RGB边缘信息指导灰度图的滤波
        gray_input = x.mean(dim=1)  # (B, H, W) 灰度图
        base_layer = self.guided_filter(x, gray_input, radius=self.radius, epsilon=self.eps)  # 返回 (B, 3, H, W)

        # 计算细节层: detail = original - base (保持三通道)
        detail_layer = x - base_layer  # (B, 3, H, W)
        
        return detail_layer


#------------------------------光谱特征提取分支--------------------------------
class SpectralBranch(nn.Module):

    def __init__(self, enhancement_factor=2.0):
        super().__init__()
        self.enhancement_factor = enhancement_factor

    def rgb_to_lab(self, image):
        '''
        将RGB图像转换为简化的LAB色彩空间表示
        输入RGB图像，形状为 (B, 3, H, W)，值范围[0,1]
        输出LAB图像，形状为 (B, 3, H, W)
        '''
        # 简化的RGB到LAB转换：
        # L通道：亮度（标准灰度计算）范围[0,1]
        # A通道：红绿差异，范围[-0.5,0.5]
        # B通道：蓝黄差异，范围[-0.5,0.5]
        R = image[:, 0:1, :, :]  # (B, 1, H, W)
        G = image[:, 1:2, :, :]  # (B, 1, H, W)
        B = image[:, 2:3, :, :]  # (B, 1, H, W)

        # L通道：标准亮度计算
        L = 0.299 * R + 0.587 * G + 0.114 * B  # 范围[0,1]

        # A通道：红绿对比
        A = (R - G) * 0.5  # 范围[-0.5,0.5]

        # B通道：蓝黄对比，修正计算
        yellow = (R + G) * 0.5  # 黄色分量
        B_ch = (B - yellow) * 0.5  # 蓝-黄差异，范围[-0.5,0.5]

        lab_image = torch.cat([L, A, B_ch], dim=1)  # (B, 3, H, W)

        return lab_image

    def apply_spectral_enhancement(self, lab_image):
        '''
        对LAB图像的A和B通道进行增强。
        输入: lab_image (B, 3, H, W) - L[0,1], A[-0.5,0.5], B[-0.5,0.5]
        '''
        # 分离 L, A, B 通道
        L = lab_image[:, 0:1, :, :]  # (B, 1, H, W)
        A = lab_image[:, 1:2, :, :]  # (B, 1, H, W)
        B = lab_image[:, 2:3, :, :]  # (B, 1, H, W)

        # 对 A 和 B 通道进行增强
        A_enhanced = A * self.enhancement_factor
        B_enhanced = B * self.enhancement_factor

        # 裁剪到合理范围，保持与原始范围一致
        A_enhanced = torch.clamp(A_enhanced, -0.5, 0.5)
        B_enhanced = torch.clamp(B_enhanced, -0.5, 0.5)

        # 重新组合
        enhanced_lab = torch.cat([L, A_enhanced, B_enhanced], dim=1)  # (B, 3, H, W)

        return enhanced_lab

    def forward(self, x):
        '''
        输入的x是经过ToTensor()后的，值范围[0,1]，形状为(B, 3, H, W)
        输出: 处理后的LAB特征图 (B, 3, H, W)
        '''

        # 直接使用输入，输入已经是[0,1]范围
        lab_image = self.rgb_to_lab(x)
        enhanced_lab = self.apply_spectral_enhancement(lab_image)

        return enhanced_lab


#------------------------------传输矩阵估计分支--------------------------------
class TransmissionBranch(nn.Module):

    def __init__(self, omega, patch_size, guided_radius, guided_eps):
        super().__init__()
        self.omega = omega
        self.patch_size = patch_size
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps

    def calculate_dark_channel(self, image, patch_size=5):
        '''
        计算暗通道图像

        参数:
        - image: 输入图像 (B, 3, H, W)
        - patch_size: 局部窗口大小，设置为5

        返回:
        - dark_channel: 暗通道图像 (B, H, W) 或 (H, W)
        '''

        min_channel = torch.min(image, dim=1)[0]  # (B, H, W)

        # 使用最大池化操作计算局部最小值 min(x) = -max(-x)
        padding = patch_size // 2  # padding = (patch_size - 1) / 2
        dark_channel = -F.max_pool2d(-min_channel.unsqueeze(1), kernel_size=patch_size, stride=1, padding=padding).squeeze(1)  # (B, H, W)
        
        return dark_channel

    def estimate_atmosphere_light(self, image, dark_channel, top_k=0.001):
        '''
        估计全局大气光值

        参数:
        - image: 输入图像 (B, 3, H, W)
        - dark_channel: 暗通道图像 (B, H, W)
        - top_k: 选择最亮像素的比例，默认值为0.001 前0.1%

        返回:
        - atmosphere_light: 全局大气光值 (B, 3) 或 (3,)
        '''

        B, C, H, W = image.shape

        # 选择暗通道中最亮的像素
        num_pixels = max(1, int(H * W * top_k))  # 至少选择1个像素
        flat_dark = dark_channel.view(B, -1)  # (B, H*W)
        _, indices = torch.topk(flat_dark, k=num_pixels, dim=1)  # (B, num_pixels)

        # 获取对应像素值
        flat_image = image.view(B, C, -1)  # (B, 3, H*W)
        batch_indices = indices.unsqueeze(1).expand(-1, C, -1)  # (B, 3, num_pixels)
        selected_pixels = torch.gather(flat_image, 2, batch_indices)  # (B, 3, num_pixels)

        # 对每个通道取最大值
        atmosphere_light = torch.max(selected_pixels, dim=2)[0]  # (B, 3)
        
        return atmosphere_light

    def calculate_transmission_map(self, dark_channel, atmosphere_light, omega=0.95):
        '''
        计算传输矩阵 t
        传输矩阵是大气消光系数与场景深度乘积的负指数函数，能反映能见度
        
        dark_channel: 暗通道图像 (B, H, W)
        atmosphere_light: 全局大气光值 (B, 3)
        omega: 保留雾的程度，默认值为0.95

        返回:
        - transmission_map: 传输矩阵 t (B, H, W)
        '''

        B, H, W = dark_channel.shape
        C = atmosphere_light.shape[1]

        # 计算每个通道的传输矩阵
        transmission_maps = []
        for c in range(C):
            # 对每个通道计算 t = 1 - ω × (dark_channel / A^c)
            A_c = atmosphere_light[:, c:c+1, None]  # (B, 1, 1)
            t_c = 1 - omega * (dark_channel / A_c)  # (B, H, W)
            transmission_maps.append(t_c)
        
        # 对所有通道的传输矩阵取平均
        transmission_map = torch.stack(transmission_maps, dim=1).mean(dim=1)  # (B, H, W)
        
        # 限制数值范围 0~1
        transmission_map = torch.clamp(transmission_map, 0, 1)
        
        return transmission_map

    def guided_filter(self, guide_img, input_img, radius=5, epsilon=1e-3):
        '''
        导向滤波，对transmission_map使用，优化结果
        guide_img使用原图
        input_img使用transmission_map

        参数:
        - guide: 导向图像 (B, 3, H, W)
        - input_img: 输入图像 (B, H, W) 或 (B, C, H, W)
        - radius: 滤波窗口半径
        - epsilon: 正则化参数

        返回:
        - output: 滤波后的图像 (B, H, W)
        '''
        B, C, H, W = guide_img.shape
        input_img = input_img.unsqueeze(1)  # (B, 1, H, W)
        input_img = input_img.expand(B, C, H, W)  # (B, 1, H, W) -> (B, 3, H, W)

        # 计算均值 使用平均池化实现box_filter的效果
        mean_guide = F.avg_pool2d(guide_img, kernel_size=2 * radius + 1, stride=1, padding=radius)  # (B, 3, H, W)
        mean_input = F.avg_pool2d(input_img, kernel_size=2 * radius + 1, stride=1, padding=radius)  # (B, C, H, W)
        mean_guide_input = F.avg_pool2d((guide_img * input_img), kernel_size=2 * radius + 1, stride=1, padding=radius)  # (B, C, H, W)
        mean_guide_sq = F.avg_pool2d((guide_img * guide_img), kernel_size=2 * radius + 1, stride=1, padding=radius)  # (B, 3, H, W)

        # 计算协方差和方差
        cov_guide_input = mean_guide_input - mean_guide * mean_input  # (B, C, H, W)
        var_guide = mean_guide_sq - mean_guide * mean_guide  # (B, 3, H, W)

        # 计算线性系数
        a = cov_guide_input / (var_guide + epsilon)  # (B, C, H, W)
        b = mean_input - a * mean_guide  # (B, C, H, W)

        # 对系数进行均值滤波 对应计算像素所在区域的所有a,b的均值
        mean_a = F.avg_pool2d(a, kernel_size=2 * radius + 1, stride=1, padding=radius)  # (B, C, H, W)
        mean_b = F.avg_pool2d(b, kernel_size=2 * radius + 1, stride=1, padding=radius)  # (B, C, H, W)

        # 滤波结果
        output = mean_a * guide_img + mean_b  # (B, C, H, W)
        # 由于input_img是单通道扩展的，所以3个通道结果相同，直接取第一个通道
        output = output[:, 0, :, :]  # (B, H, W)
        
        return output

    def forward(self, x):
        '''
        输入 x_rgb_tensor: PyTorch Tensor (B, 3, H, W)，经过ToTensor()后的，值范围[0,1]
        输出: 传输矩阵 (B, 1, H, W)
        '''
        # 直接使用输入，输入已经是[0,1]范围
        # 计算暗通道图像
        dark_channel = self.calculate_dark_channel(x, self.patch_size)  # (B, H, W)

        # 估计全局大气光值
        atmosphere_light = self.estimate_atmosphere_light(x, dark_channel)  # (B, 3)

        # 计算传输矩阵
        transmission_map = self.calculate_transmission_map(dark_channel, atmosphere_light, self.omega)  # (B, H, W)

        # 导向滤波优化传输矩阵
        refined_transmission_map = self.guided_filter(
            x,  # 导向图：RGB图像 (B, 3, H, W)
            transmission_map,  # 输入图：传输图 (B, H, W)
            radius=self.guided_radius,
            epsilon=self.guided_eps)  # (B, H, W)

        # 添加通道维度，输出 (B, 1, H, W)
        refined_transmission_map = refined_transmission_map.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        return refined_transmission_map


#------------------------------特征提取块--------------------------------
def feature_extraction_block(ori_inputs, aug_inputs):
    '''
    ori_inputs: (B, 3, H, W) 原始RGB tensor，用于物理特征提取
    aug_inputs: (B, 3, H, W) 增强后RGB tensor，用于特征融合

    输出：
        init_features: (B, C, H, W) 或 (C, H, W) 拼接后的特征tensor
        num_channels: 拼接后的特征tensor的通道数，通道组成：深度(16 or 1) + 透射图(1) + 光谱(3) + 细节(3) + RGB(3)
    '''
    # 所有分支都在GPU上计算
    if config.USE_SIMPLE_DEPTH:
        scene_depth_branch = SimpleSceneDepthBranch(model_weight_path=config.SIMPLE_DEPTH_MODEL_PATH, device=config.DEVICE)
    else:
        scene_depth_branch = DPTSceneDepthBranch(dpt_model_type='dpt_hybrid', device=config.DEVICE)

    transmission_branch = TransmissionBranch(omega=config.TRANSMISSION_OMEGA,
                                            patch_size=config.TRANSMISSION_PATCH_SIZE,
                                            guided_radius=config.TRANSMISSION_GUIDED_RADIUS,
                                            guided_eps=config.TRANSMISSION_GUIDED_EPS).to(config.DEVICE)

    spectral_branch = SpectralBranch(enhancement_factor=config.SPECTRAL_ENHANCEMENT_FACTOR).to(config.DEVICE)

    detail_branch = DetailBranch(guided_radius=config.DETAIL_GUIDED_RADIUS, guided_eps=config.DETAIL_GUIDED_EPS).to(config.DEVICE)

    # 提取各分支特征，所有计算都在GPU上
    depth_feat = scene_depth_branch(ori_inputs)  # (B, 16, H, W) 或 (B, 1, H, W)
    transmission_feat = transmission_branch(ori_inputs)  # (B, 1, H, W)
    spectral_feat = spectral_branch(ori_inputs)  # (B, 3, H, W)
    detail_feat = detail_branch(ori_inputs)  # (B, 3, H, W)
    rgb_feat = aug_inputs  # (B, 3, H, W)
    
    # 拼接特征
    init_features = torch.cat([rgb_feat, depth_feat, transmission_feat, spectral_feat, detail_feat], dim=1)  # (B, C, H, W)
    num_channels = init_features.shape[1]
    
    return init_features, num_channels