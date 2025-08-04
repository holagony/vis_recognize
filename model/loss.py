
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from utils import config

class FocalLoss(nn.Module):
    """
    改进的Focal Loss实现，特别针对类别不平衡问题优化
    """
    def __init__(self, alpha=1, gamma=2, weight=None, size_average=True, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # 使用标签平滑的CrossEntropyLoss
        ce_loss = nn.CrossEntropyLoss(
            weight=self.weight, 
            reduction='none', 
            label_smoothing=self.label_smoothing)(inputs, targets)
        
        # 计算概率
        pt = torch.exp(-ce_loss)
        
        # 自适应alpha：对困难样本使用更高的alpha
        if isinstance(self.alpha, (int, float)):
            alpha_t = self.alpha
        else:
            # 如果alpha是张量，根据类别选择对应的alpha值
            alpha_t = self.alpha.gather(0, targets)
        
        # Focal Loss公式
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def create_loss_function(labels, loss_type='crossentropy', alpha=1, gamma=2, use_weights=False, 
                         weight_mode='balanced', smooth_factor=0.1, label_smoothing=0.0):
    """
    创建损失函数 - 支持CrossEntropyLoss和FocalLoss二选一
    
    Args:
        labels: 训练标签
        loss_type: 损失函数类型 ('crossentropy' 或 'focal')
        alpha: Focal Loss的alpha参数，可以是数值或'auto'
        gamma: Focal Loss的gamma参数
        use_weights: 是否在损失函数中使用类别权重，选择crossentropy的时候开启
        weight_mode: 权重计算模式 ('balanced', 'sqrt_balanced', 'log_balanced')
        smooth_factor: 权重平滑因子，减少极端权重
        label_smoothing: 标签平滑参数，有助于提高泛化能力
    """
    weights_tensor = None
    
    if use_weights:
        label_counts = Counter(labels)
        unique_labels = np.unique(labels)
        total_samples = len(labels)
        
        if weight_mode == 'balanced':
            class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)

        elif weight_mode == 'sqrt_balanced':
            # 使用平方根倒数，减少极端权重
            class_weights = []
            for label in unique_labels:
                count = label_counts[label]
                weight = np.sqrt(total_samples / (len(unique_labels) * count))
                class_weights.append(weight)
            class_weights = np.array(class_weights)

        elif weight_mode == 'log_balanced':
            # 使用对数平滑的权重
            class_weights = []
            for label in unique_labels:
                count = label_counts[label]
                weight = np.log(total_samples / count + 1)
                class_weights.append(weight)
            class_weights = np.array(class_weights)
        
        # 权重平滑：w_smooth = (1-smooth_factor) * w + smooth_factor * 1.0
        class_weights = (1 - smooth_factor) * class_weights + smooth_factor * np.ones_like(class_weights)
        
        # 处理缺失的类别
        weight_dict = dict(zip(unique_labels, class_weights))
        final_weights = []
        for i in range(config.NUM_CLASSES):
            if i in weight_dict:
                final_weights.append(weight_dict[i])
            else:
                final_weights.append(1.0)
        
        weights_tensor = torch.FloatTensor(final_weights).to(config.DEVICE)
        print(f"类别权重 ({weight_mode}模式): {final_weights}")
    
    if loss_type.lower() == 'focal':
        # 为Focal Loss计算类别特定的alpha值
        if alpha == 'auto' and use_weights:
            # 使用类别权重作为alpha值，给少数类别更高的关注
            alpha_values = weights_tensor if weights_tensor is not None else 1.0
        else:
            alpha_values = alpha
            
        return FocalLoss(
            alpha=alpha_values, 
            gamma=gamma, 
            weight=weights_tensor,
            label_smoothing=label_smoothing)
    else:  # CrossEntropyLoss
        return nn.CrossEntropyLoss(
            weight=weights_tensor,
            label_smoothing=label_smoothing)