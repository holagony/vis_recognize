
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from utils import config

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance
    """
    def __init__(self, alpha=1, gamma=2, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def create_loss_function(labels, loss_type='crossentropy', alpha=1, gamma=2, use_weights=True):
    """
    创建损失函数 - 支持CrossEntropyLoss和FocalLoss二选一
    
    Args:
        labels: 训练标签
        loss_type: 损失函数类型 ('crossentropy' 或 'focal')
        alpha: Focal Loss的alpha参数
        gamma: Focal Loss的gamma参数
        use_weights: 是否在损失函数中使用类别权重 加权
    """
    weights_tensor = None
    
    if use_weights:
        # 计算类别权重
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
        
        # 处理缺失的类别
        weight_dict = dict(zip(unique_labels, class_weights))
        final_weights = []
        for i in range(config.NUM_CLASSES):
            if i in weight_dict:
                final_weights.append(weight_dict[i])
            else:
                final_weights.append(1.0)
        
        weights_tensor = torch.FloatTensor(final_weights).to(config.DEVICE)
    
    if loss_type.lower() == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma, weight=weights_tensor)
    else:  # CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weights_tensor)