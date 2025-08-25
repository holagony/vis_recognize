import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from utils import config


class FocalLoss(nn.Module):

    def __init__(self, alpha, gamma, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        # Focal Loss 不使用 weight 参数，只使用 alpha 来调整类别权重
        # 避免与 alpha 权重产生冲突
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)

        # 计算概率：pt = p_t if y=1 else 1-p_t
        pt = torch.exp(-ce_loss) # 对于正确类别的概率
        pt = torch.clamp(pt, min=1e-7, max=1.0) # 确保数值稳定性

        alpha_tensor = self.alpha[targets].to(inputs.device)
        focal_loss = alpha_tensor * (1 - pt)**self.gamma * ce_loss

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class DiceCELoss(nn.Module):
    """
    组合损失函数：Dice Loss + Cross Entropy Loss
    结合了Dice Loss的边界优化和Cross Entropy的分类能力
    """
    
    def __init__(self, 
                 dice_weight=0.5,
                 ce_weight=0.5,
                 dice_smooth=1e-6,
                 ce_label_smoothing=0.0,
                 use_class_weights=False,
                 class_weights=None):
        """
        Args:
            dice_weight: Dice Loss的权重
            ce_weight: Cross Entropy Loss的权重
            dice_smooth: Dice Loss的平滑因子
            ce_label_smoothing: Cross Entropy的标签平滑
            use_class_weights: 是否在Cross Entropy中使用类别权重
            class_weights: 类别权重张量
        """
        super(DiceCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_smooth = dice_smooth
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights if use_class_weights else None,
            label_smoothing=ce_label_smoothing,
            reduction='mean'
        )
    
    def forward(self, inputs, targets):
        # 计算Cross Entropy Loss
        ce_loss = self.ce_loss(inputs, targets)
        
        # 计算Dice Loss
        dice_loss = self._dice_loss(inputs, targets)
        
        # 组合损失
        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        
        return total_loss
    
    def _dice_loss(self, inputs, targets):
        """计算Dice Loss"""
        # 应用softmax获取概率
        probs = torch.softmax(inputs, dim=1)
        
        # 对于分类任务，直接使用概率和目标
        # 创建one-hot编码的目标
        targets_onehot = F.one_hot(targets, num_classes=probs.size(1)).float()
        # 对于分类任务，我们计算每个类别的Dice系数
        # 将形状调整为 [N, C, 1] 以便统一处理
        probs = probs.unsqueeze(-1)  # [N, C] -> [N, C, 1]
        targets_onehot = targets_onehot.unsqueeze(-1)  # [N, C] -> [N, C, 1]
        
        # 计算Dice系数
        intersection = (probs * targets_onehot).sum(dim=2)
        denominator = probs.sum(dim=2) + targets_onehot.sum(dim=2)
        
        # 计算Dice系数
        dice = (2.0 * intersection + self.dice_smooth) / (denominator + self.dice_smooth)
        
        # 计算Dice Loss
        dice_loss = 1 - dice
        
        return dice_loss.mean()


def create_loss_function(labels, 
                         loss_type='crossentropy', 
                         use_weights=False, 
                         weight_mode='balanced', 
                         weight_smoothing=False, 
                         smooth_factor=0.05, 
                         label_smoothing=0.05):
    '''
    创建损失函数 - 支持CrossEntropyLoss、FocalLoss和DiceCELoss
    labels: 训练标签
    loss_type: 损失函数类型 ('crossentropy', 'focal', 'dice_ce')
    use_weights: 是否在损失函数中使用类别权重
    weight_mode: 权重(alpha)的计算模式 ('balanced', 'sqrt_balanced')
    weight_smoothing: 是否对计算出的类别权重进行平滑处理
    smooth_factor: 权重平滑因子，减少极端权重（仅当weight_smoothing=True时有效）
    label_smoothing: crossentropy的时候使用
    '''

    # 计算类别权重（如果需要）
    weights_tensor = None
    alpha_tensor = None
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

        # 权重平滑处理 - 可选择性平滑策略
        if len(class_weights) > 0 and weight_smoothing:
            # 计算权重统计信息
            weight_mean = np.mean(class_weights)
            weight_std = np.std(class_weights)

            # 只对极端权重进行平滑，避免过度平滑
            if weight_std > 1e-8:  # 避免数值不稳定
                # 使用基于标准差的阈值来识别极端值
                upper_threshold = weight_mean + 2 * weight_std
                lower_threshold = weight_mean - 2 * weight_std

                smoothed_weights = []
                for weight in class_weights:
                    if weight > upper_threshold:
                        # 对过高的权重进行平滑
                        smoothed_weight = upper_threshold * (1 - smooth_factor) + weight_mean * smooth_factor
                    elif weight < lower_threshold:
                        # 对过低的权重进行平滑
                        smoothed_weight = lower_threshold * (1 - smooth_factor) + weight_mean * smooth_factor
                    else:
                        # 保持正常权重不变
                        smoothed_weight = weight
                    smoothed_weights.append(smoothed_weight)

                class_weights = np.array(smoothed_weights)

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

        # 为focal loss计算每个类别的alpha值
        if loss_type.lower() == 'focal':
            alpha_values = np.array(final_weights)
            
            # 使用Min-Max归一化到[0.1, 1.0]范围，保持相对比例
            if alpha_values.max() > alpha_values.min():
                alpha_values = 0.1 + 0.9 * (alpha_values - alpha_values.min()) / (alpha_values.max() - alpha_values.min())
            else:
                alpha_values = np.full_like(alpha_values, 0.5)
            
            alpha_tensor = torch.FloatTensor(alpha_values).to(config.DEVICE)

    # 根据损失类型返回相应的损失函数
    if loss_type.lower() == 'focal':
        # 优先使用计算出的alpha_tensor，如果没有则使用配置的FOCAL_ALPHA
        if alpha_tensor is not None:
            focal_alpha = alpha_tensor
        else:
            alpha = np.array(config.FOCAL_ALPHA)
            focal_alpha = torch.FloatTensor(alpha).to(config.DEVICE)
        return FocalLoss(alpha=focal_alpha, gamma=config.FOCAL_GAMMA)
    
    elif loss_type.lower() == 'dice_ce':
        return DiceCELoss(
            dice_weight=config.DICE_WEIGHT,
            ce_weight=config.CE_WEIGHT,
            dice_smooth=config.DICE_SMOOTH,
            ce_label_smoothing=label_smoothing,
            use_class_weights=use_weights,
            class_weights=weights_tensor)
    
    else:  # crossentropy (默认)
        return nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)


def supcon_loss(features, labels, temperature):
    '''
    Supervised Contrastive Loss (InfoNCE-based)
    features: [batch_size, projection_dim]
    labels: [batch_size]
    '''
    batch_size = features.shape[0]
    # 归一化特征
    features = F.normalize(features, dim=1)
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # 创建标签掩码：同一类的样本为正样本
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(config.DEVICE)
    
    # 去除自比较
    mask_self = torch.eye(batch_size, dtype=torch.bool).to(config.DEVICE)
    mask = mask.masked_fill(mask_self, 0)
    
    # 计算正样本和负样本的相似度
    exp_sim = torch.exp(similarity_matrix) * (1 - mask_self.float())
    pos_sum = torch.sum(exp_sim * mask, dim=1, keepdim=True)
    neg_sum = torch.sum(exp_sim * (1 - mask), dim=1, keepdim=True)
    
    # 避免除零
    pos_sum = torch.clamp(pos_sum, min=1e-9)
    neg_sum = torch.clamp(neg_sum, min=1e-9)
    
    # 计算损失
    loss = -torch.log(pos_sum / (pos_sum + neg_sum))
    loss = loss.mean()
    
    return loss