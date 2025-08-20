import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from utils import config


class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, weight=None, size_average=True, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none', label_smoothing=self.label_smoothing)(inputs, targets)
        
        # 计算概率：pt = p_t if y=1 else 1-p_t
        # 对于正确类别的概率
        pt = torch.exp(-ce_loss)
        # 确保数值稳定性
        pt = torch.clamp(pt, min=1e-7, max=1.0)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def create_loss_function(labels, loss_type='crossentropy', alpha=1, gamma=2, use_weights=False, weight_mode='balanced', smooth_factor=0.05, label_smoothing=0.1):
    '''
    创建损失函数 - 支持CrossEntropyLoss和FocalLoss二选一
    labels: 训练标签
    loss_type: 损失函数类型 ('crossentropy' 或 'focal')
    use_weights: 是否在损失函数中使用类别权重（仅CrossEntropyLoss有效）
    weight_mode: 权重计算模式 ('balanced', 'sqrt_balanced')
    smooth_factor: 权重平滑因子，减少极端权重
    label_smoothing: 标签平滑参数，有助于提高泛化能力
    '''
    if loss_type.lower() == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma, weight=None, label_smoothing=label_smoothing)  # weight=None 不使用额外权重，避免冲突

    else:
        if use_weights:
            label_counts = Counter(labels)
            unique_labels = list(label_counts.keys())
            total_samples = len(labels)

            if weight_mode == 'balanced':
                class_weights = compute_class_weight('balanced', classes=np.array(unique_labels), y=np.array(labels))

            elif weight_mode == 'sqrt_balanced':
                class_weights = []
                for label in unique_labels:
                    count = label_counts[label]
                    weight = np.sqrt(total_samples / (len(unique_labels) * count))
                    class_weights.append(weight)
                class_weights = np.array(class_weights)

            # 权重平滑处理
            if len(class_weights) > 0:
                weight_range = class_weights.max() - class_weights.min()
                if weight_range > 1e-8:  # 避免数值不稳定
                    normalized_weights = (class_weights - class_weights.min()) / weight_range
                    mean_weight = normalized_weights.mean()
                    class_weights = (1 - smooth_factor) * normalized_weights + smooth_factor * mean_weight


            # 缺失类别处理
            weight_dict = dict(zip(unique_labels, class_weights))            
            default_weight = np.mean(class_weights) if len(class_weights) > 0 else 1.0
            final_weights = []

            for i in range(config.NUM_CLASSES):
                if i in weight_dict:
                    final_weights.append(weight_dict[i])
                else:
                    final_weights.append(default_weight)

            # 创建权重张量并移动到指定设备
            weights_tensor = torch.FloatTensor(final_weights).to(config.DEVICE)
        else:
            weights_tensor = None

        return nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)
