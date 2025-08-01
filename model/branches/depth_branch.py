import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dpt.dpt.models import DPTDepthModel
from torchvision.models import mobilenet_v2
from torchvision.transforms import functional as TF

class DPTSceneDepthBranch(nn.Module):
    '''
    DPT模型内部固定使用384x384，但外部可以接受任意尺寸，会自动调整到384x384
    DPT模型可选"dpt_large" or "dpt_hybrid"
    输入：RGB图像，形状为(B, 3, H, W)
    输出：深度图，形状为(B, 1, H, W)
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
        for param in self.dpt_model.parameters(): # 冻结参数
            param.requires_grad = False
        
        # 将整个模块移动到指定设备
        self.to(self.device)
        self.dpt_model.eval()

    def forward(self, x):
        with torch.no_grad():
            depth_map = self.dpt_model(x)
        
        # 将深度图调整回原始输入尺寸
        original_size = x.shape[2:]
        if depth_map.shape[2:] != original_size:
            depth_map = F.interpolate(depth_map.unsqueeze(1), size=original_size, mode='bicubic', align_corners=False)
        else:
            depth_map = depth_map.unsqueeze(1)
        
        return depth_map


class MobileNetEncoder(nn.Module):
    def __init__(self, device='cpu'): # 移除 model_weight_path 参数
        super().__init__()
        self.device = device # 保存 device

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
        x0 = self.layer0(x); skips['skip1'] = x0
        x1 = self.layer1(x0); skips['skip2'] = x1
        x2 = self.layer2(x1); skips['skip3'] = x2
        x3 = self.layer3(x2); skips['skip4'] = x3
        bottleneck = self.layer4(x3)
        return bottleneck, skips


class MobileNetDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=5):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, skip_feature):
        x = self.upsample(x)
        if x.shape[2:] != skip_feature.shape[2:]:
             x = TF.resize(x, skip_feature.shape[2:])
        x = torch.cat([x, skip_feature], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SimpleSceneDepthBranch(nn.Module):
    def __init__(self, 
                 model_weight_path, 
                 freeze_weights=True,
                 device='cpu'):
        super().__init__()
        self.device = device # 保存 device

        # 创建编码器（不加载权重）
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
                print(f"成功加载权重文件: {model_weight_path}")
            except RuntimeError as e:
                print(f"权重加载失败: {e}")
                print("尝试非严格模式加载...")
                self.load_state_dict(state_dict, strict=False)

        if freeze_weights:
            for param in self.parameters():
                param.requires_grad = False
        
        self.to(self.device)

    def forward(self, x):
        # x = x.to(self.device) # 输入tensor应该在进入模型前被移动到device
        bottleneck, skips = self.encoder(x)
        d4 = self.dec4(bottleneck, skips['skip4'])
        d3 = self.dec3(d4, skips['skip3'])
        d2 = self.dec2(d3, skips['skip2'])
        d1 = self.dec1(d2, skips['skip1'])
        features_before_final_conv = self.final_upsample(d1)
        # final_conv = self.final_conv(features_before_final_conv)
        return features_before_final_conv