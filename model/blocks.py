import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention_modules import CBAM

class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积，减少参数量
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                 stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x

class SFRB(nn.Module):
    """
    浅层特征表示块
    """
    def __init__(self, in_channels, out_channels):
        super(SFRB, self).__init__()

        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels//2)
        self.conv2 = DepthwiseSeparableConv(out_channels//2, out_channels)
        
        # 残差连接的维度调整
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                          nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)      # 已包含: Conv->BN->ReLU->Conv->BN->ReLU
        out = self.conv2(out)    # 已包含: Conv->BN->ReLU->Conv->BN->ReLU
        out += identity          # 残差连接
        
        return out

class PyramidPooling(nn.Module):
    """
    金字塔池化模块
    """
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes
        self.pools = nn.ModuleList()
        
        for pool_size in pool_sizes:
            self.pools.append(nn.Sequential(nn.AdaptiveAvgPool2d(pool_size),
                                            nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1, bias=False),
                                            nn.BatchNorm2d(in_channels // len(pool_sizes)),
                                            nn.LeakyReLU()))
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(in_channels),
                                  nn.LeakyReLU())

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pool_outs = [x]
        
        for pool in self.pools:
            pooled = pool(x)
            pooled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            pool_outs.append(pooled)
        
        out = torch.cat(pool_outs, dim=1)
        out = self.conv(out)
        
        return out

class MSFB(nn.Module):
    """
    多尺度融合块
    """
    def __init__(self, channels, dilations=[1, 2, 3, 5]):
        super(MSFB, self).__init__()
        self.dilations = dilations
        
        # 多尺度空洞卷积分支
        self.dilated_convs = nn.ModuleList()
        for dilation in dilations:
            self.dilated_convs.append(nn.Sequential(nn.Conv2d(channels, channels//len(dilations), 3, padding=dilation, dilation=dilation, bias=False),
                                      nn.BatchNorm2d(channels//len(dilations)),
                                      nn.LeakyReLU()))
        # 1x1卷积降维
        self.reduce_conv = nn.Sequential(nn.Conv2d(channels, channels//4, 1, bias=False),
                                         nn.BatchNorm2d(channels//4),
                                         nn.LeakyReLU())
        # 金字塔池化
        self.pyramid_pool = PyramidPooling(channels//4)
        
        # 特征融合 - 确保输出通道数与输入一致
        fusion_input_channels = channels + channels//4  # 多尺度特征 + 金字塔特征
        self.fusion_conv = nn.Sequential(nn.Conv2d(fusion_input_channels, channels, 3, padding=1, bias=False),
                                         nn.BatchNorm2d(channels),
                                         nn.LeakyReLU())

        # CBAM注意力机制
        self.cbam_attention = CBAM(channels)

    def forward(self, x):
        # 多尺度空洞卷积
        dilated_feats = []
        for conv in self.dilated_convs:
            dilated_feats.append(conv(x))
        
        # 拼接多尺度特征
        multi_scale = torch.cat(dilated_feats, dim=1)
        
        # 降维和金字塔池化
        reduced = self.reduce_conv(x)
        pyramid_feat = self.pyramid_pool(reduced)
        
        # 特征融合
        combined = torch.cat([multi_scale, pyramid_feat], dim=1)
        output = self.fusion_conv(combined)
        
        # 残差连接 - 现在输入输出维度一致
        residual_output = x + output
        
        # CBAM注意力
        attended_output = self.cbam_attention(residual_output)
        
        return attended_output

class GFFB(nn.Module):
    """
    全局特征融合块
    """
    def __init__(self, num_msfb_outputs, channels_per_msfb, out_channels):
        super(GFFB, self).__init__()
        
        # 使用简单的特征拼接
        fusion_input_channels = num_msfb_outputs * channels_per_msfb # 2*128
        
        # 特征增强
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(fusion_input_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU())
        
        # CBAM注意力
        self.cbam_attention = CBAM(out_channels)
        
        # 残差连接的维度调整（如果需要）
        if fusion_input_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(fusion_input_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, msfb_outputs_list):
        # 直接使用特征拼接
        fused_features = torch.cat(msfb_outputs_list, dim=1)
        
        # 保存用于残差连接
        identity = self.shortcut(fused_features)
        
        # 特征增强
        enhanced_features = self.feature_enhance(fused_features)
        
        # CBAM注意力
        attended_features = self.cbam_attention(enhanced_features)
        
        # 残差连接
        output = attended_features + identity
        
        return output