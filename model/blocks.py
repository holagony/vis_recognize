import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention_modules import CBAM

class SFRB(nn.Module):
    """
    浅层特征表示块
    """
    def __init__(self, in_channels, out_channels):
        super(SFRB, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),  # 1x1卷积精炼特征
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU())
        
        # 残差连接的维度调整
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                          nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        
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
    多尺度融合块 - 支持渐进式通道增长
    """
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 3, 5]):
        super(MSFB, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 多尺度空洞卷积分支
        self.dilated_convs = nn.ModuleList()
        for dilation in dilations:
            self.dilated_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels//len(dilations), 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels//len(dilations)),
                nn.LeakyReLU()))
        
        # 1x1卷积降维用于金字塔池化
        pyramid_channels = max(16, out_channels//4)
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, pyramid_channels, 1, bias=False),
            nn.BatchNorm2d(pyramid_channels),
            nn.LeakyReLU())
        
        # 金字塔池化
        self.pyramid_pool = PyramidPooling(pyramid_channels)
        
        # 特征融合 - 支持不同的输入输出通道数
        fusion_input_channels = out_channels + pyramid_channels  # 多尺度特征 + 金字塔特征
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_input_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU())

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

        # CBAM注意力机制
        self.cbam_attention = CBAM(out_channels)

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
        
        # 残差连接 - 支持不同维度
        identity = self.shortcut(x)
        residual_output = identity + output
        
        # CBAM注意力
        attended_output = self.cbam_attention(residual_output)
        
        return attended_output

class GFFB(nn.Module):
    """
    全局特征融合块 - 支持不同通道数的特征融合
    """
    def __init__(self, in_channels, out_channels):
        super(GFFB, self).__init__()
        
        # 先降维到中间通道数，再升维到输出通道数
        mid_channels = min(in_channels, out_channels) // 2
        
        # 特征增强 - 两阶段设计
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),  # 1x1精炼
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU())
        
        # CBAM注意力
        self.cbam_attention = CBAM(out_channels)
        
        # 残差连接的维度调整
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, fused_features):
        identity = self.shortcut(fused_features)
        enhanced_features = self.feature_enhance(fused_features)
        attended_features = self.cbam_attention(enhanced_features)
        output = attended_features + identity
        
        return output