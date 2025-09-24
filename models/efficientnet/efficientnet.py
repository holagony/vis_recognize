import torch
import torch.nn as nn
import timm
from typing import Optional


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet编码器，支持多通道输入和特征提取
    """
    
    def __init__(self, model_name='efficientnet_b0', in_channels=11, num_classes=5, pretrained=False):
        super(EfficientNetEncoder, self).__init__()
        
        self.model_name = model_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # 创建EfficientNet模型
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # 移除分类头，只保留特征提取
            global_pool='avg'  # 使用平均池化
        )
        
        # 获取特征维度
        self.feature_dim = self.backbone.num_features
        
        # 如果输入通道不是3，需要修改第一层
        if in_channels != 3:
            self._modify_first_conv(in_channels)
        
        # 添加分类头
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _modify_first_conv(self, in_channels):
        """
        修改第一个卷积层以支持多通道输入
        """
        # 获取原始第一层卷积
        first_conv = self.backbone.conv_stem
        
        # 创建新的卷积层
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        
        # 替换第一层
        self.backbone.conv_stem = new_conv
    
    def _initialize_weights(self):
        """
        权重初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x):
        """
        提取特征表示，不包含最后的分类层
        用于对比学习
        """
        features = self.backbone(x)  # [B, feature_dim]
        return features
    
    def forward(self, x):
        """
        前向传播
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits


class EfficientNetJointModel(nn.Module):
    """
    EfficientNet + SupCon联合模型
    支持对比学习和分类任务
    """
    
    def __init__(self, model_name='efficientnet_b0', in_channels=11, 
                 projection_dim=128, num_classes=5, pretrained=False):
        super(EfficientNetJointModel, self).__init__()
        
        # 创建编码器
        self.encoder = EfficientNetEncoder(
            model_name=model_name,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained
        )
        
        # 获取特征维度
        self.feature_dim = self.encoder.feature_dim
        
        # 投影头：用于对比学习
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, projection_dim)
        )
        
        # 分类头：用于交叉熵损失
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # 权重初始化
        self._initialize_projector()
    
    def _initialize_projector(self):
        """
        初始化投影头权重
        """
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 初始化分类头
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        返回: (features, projections, logits)
        """
        # 提取特征
        h = self.encoder.extract_features(x)  # [B, feature_dim]
        
        # 投影到对比学习空间
        z = self.projector(h)  # [B, projection_dim]
        
        # 分类输出
        logits = self.classifier(h)  # [B, num_classes]
        
        return h, z, logits


# 便捷函数
def efficientnet_b0(in_channels=11, num_classes=5, pretrained=False, **kwargs):
    """
    创建EfficientNet-B0模型
    """
    model = EfficientNetEncoder(
        model_name='efficientnet_b0',
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained
    )
    return model


def efficientnet_b1(in_channels=11, num_classes=5, pretrained=False, **kwargs):
    """
    创建EfficientNet-B1模型
    """
    model = EfficientNetEncoder(
        model_name='efficientnet_b1',
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained
    )
    return model


def efficientnet_b0_supcon(in_channels=11, projection_dim=128, num_classes=5, pretrained=False, **kwargs):
    """
    创建EfficientNet-B0 + SupCon模型
    """
    model = EfficientNetJointModel(
        model_name='efficientnet_b0',
        in_channels=in_channels,
        projection_dim=projection_dim,
        num_classes=num_classes,
        pretrained=pretrained
    )
    return model


def efficientnet_b1_supcon(in_channels=11, projection_dim=128, num_classes=5, pretrained=False, **kwargs):
    """
    创建EfficientNet-B1 + SupCon模型
    """
    model = EfficientNetJointModel(
        model_name='efficientnet_b1',
        in_channels=in_channels,
        projection_dim=projection_dim,
        num_classes=num_classes,
        pretrained=pretrained
    )
    return model


def efficientnet_b2_supcon(in_channels=11, projection_dim=128, num_classes=5, pretrained=False, **kwargs):
    """
    创建EfficientNet-B2 + SupCon模型（为512x512输入准备）
    """
    model = EfficientNetJointModel(
        model_name='efficientnet_b2',
        in_channels=in_channels,
        projection_dim=projection_dim,
        num_classes=num_classes,
        pretrained=pretrained
    )
    return model


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试EfficientNet-B0
    print("Testing EfficientNet-B0...")
    model_b0 = efficientnet_b0(in_channels=11, num_classes=5)
    model_b0 = model_b0.to(device)
    
    # 测试输入
    x = torch.randn(2, 11, 384, 384).to(device)
    
    with torch.no_grad():
        output = model_b0(x)
        print(f"B0 Output shape: {output.shape}")
        print(f"B0 Feature dim: {model_b0.feature_dim}")
    
    # 测试EfficientNet-B1 + SupCon
    print("\nTesting EfficientNet-B1 + SupCon...")
    model_b1_supcon = efficientnet_b1_supcon(in_channels=11, projection_dim=128, num_classes=5)
    model_b1_supcon = model_b1_supcon.to(device)
    
    with torch.no_grad():
        h, z, logits = model_b1_supcon(x)
        print(f"B1 SupCon - Features shape: {h.shape}")
        print(f"B1 SupCon - Projections shape: {z.shape}")
        print(f"B1 SupCon - Logits shape: {logits.shape}")
        print(f"B1 SupCon - Feature dim: {model_b1_supcon.feature_dim}")
    
    print("\nAll tests passed!")