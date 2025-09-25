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
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)

        # 计算概率：pt = p_t if y=1 else 1-p_t
        pt = torch.exp(-ce_loss)  # 对于正确类别的概率
        pt = torch.clamp(pt, min=1e-7, max=1.0)  # 确保数值稳定性

        alpha_tensor = self.alpha[targets].to(inputs.device)
        focal_loss = alpha_tensor * (1 - pt)**self.gamma * ce_loss

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class LogitAdjustmentLoss(nn.Module):
    '''
    Logit Adjustment Loss for Long-Tailed Recognition
    论文: Long-tail learning via logit adjustment (ICLR 2021)
    
    通过调整logits来补偿训练数据中的类别不平衡，特别适用于长尾分布数据
    '''
    
    def __init__(self, class_frequencies, tau=1.0, base_loss='crossentropy', label_smoothing=0.0, class_weights=None):
        '''
        Args:
            class_frequencies: 每个类别在训练集中的频率 [num_classes]
            tau: 调整强度参数，越大调整越强
            base_loss: 基础损失函数类型 ('crossentropy', 'focal')
            label_smoothing: 标签平滑参数
            class_weights: 类别权重（可选）
        '''
        super(LogitAdjustmentLoss, self).__init__()
        
        # 计算logit adjustment偏移量
        # adjustment = tau * log(class_frequencies)
        self.register_buffer('logit_adjustments', 
                           tau * torch.log(torch.FloatTensor(class_frequencies) + 1e-12))
        
        self.base_loss = base_loss.lower()
        
        if self.base_loss == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, 
                                               label_smoothing=label_smoothing)
        elif self.base_loss == 'focal':
            # 如果使用focal loss作为基础损失，需要传入alpha参数
            if class_weights is not None:
                alpha = class_weights / class_weights.sum() * len(class_weights)
            else:
                alpha = torch.ones(len(class_frequencies))
            self.criterion = FocalLoss(alpha=alpha, gamma=2.0)
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")
    
    def forward(self, logits, targets):
        '''
        Args:
            logits: 模型输出的logits [batch_size, num_classes]
            targets: 真实标签 [batch_size]
        '''
        # 应用logit adjustment
        # 在训练时，对logits进行调整以补偿类别不平衡
        if self.training:
            adjusted_logits = logits + self.logit_adjustments.unsqueeze(0)
        else:
            # 在推理时，可以选择是否应用adjustment
            # 通常在验证/测试时不应用，以获得更好的校准
            adjusted_logits = logits
            
        return self.criterion(adjusted_logits, targets)
    
    def set_inference_mode(self, apply_adjustment=False):
        '''
        设置推理模式下是否应用logit adjustment
        Args:
            apply_adjustment: 是否在推理时应用adjustment
        '''
        self.apply_adjustment_in_inference = apply_adjustment


class DiceCELoss(nn.Module):
    '''
    组合损失函数：Dice Loss + Cross Entropy Loss
    结合了Dice Loss的边界优化和Cross Entropy的分类能力
    dice_weight: Dice Loss的权重
    ce_weight: Cross Entropy Loss的权重
    dice_smooth: Dice Loss的平滑因子
    ce_label_smoothing: Cross Entropy的标签平滑
    use_class_weights: 是否在Cross Entropy中使用类别权重
    class_weights: 类别权重张量
    '''

    def __init__(self, dice_weight=0.5, ce_weight=0.5, dice_smooth=1e-6, ce_label_smoothing=0.0, class_weights=None):
        super(DiceCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_smooth = dice_smooth
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights , label_smoothing=ce_label_smoothing, reduction='mean')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        dice_loss = self._dice_loss(inputs, targets)
        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        return total_loss

    def _dice_loss(self, inputs, targets):
        ''' 
        计算Dice Loss
        '''
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


def create_loss_function(labels, loss_type='crossentropy', use_weights=False, weight_mode='balanced', weight_smoothing=False, smooth_factor=0.05, label_smoothing=0.05, logit_adjustment_tau=None):
    '''
    创建损失函数 - 支持CrossEntropyLoss、FocalLoss、DiceCELoss和LogitAdjustmentLoss
    labels: 训练标签
    loss_type: 损失函数类型 ('crossentropy', 'focal', 'dice_ce', 'logit_adjustment')
    use_weights: 是否在损失函数中使用类别权重，计算得到的权重用在ce loss上面
    weight_mode: 权重的计算模式 ('balanced', 'sqrt_balanced')
    weight_smoothing: 是否对计算出的类别权重进行平滑处理
    smooth_factor: 权重平滑因子，减少极端权重（仅当weight_smoothing=True时有效）
    label_smoothing: crossentropy的时候使用
    logit_adjustment_tau: logit adjustment的调整强度参数，如果为None则使用配置文件中的值
    '''

    # 计算类别权重
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

        # 权重平滑处理
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
        return DiceCELoss(dice_weight=config.DICE_WEIGHT, ce_weight=config.CE_WEIGHT, dice_smooth=config.DICE_SMOOTH, ce_label_smoothing=label_smoothing, class_weights=weights_tensor)
    
    elif loss_type.lower() == 'logit_adjustment':
        # 计算类别频率
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        # 计算每个类别的频率
        class_frequencies = []
        for i in range(config.NUM_CLASSES):
            if i in label_counts:
                frequency = label_counts[i] / total_samples
            else:
                frequency = 1e-12  # 避免log(0)
            class_frequencies.append(frequency)
        
        # 使用传入的tau参数或配置文件中的值
        tau = logit_adjustment_tau if logit_adjustment_tau is not None else config.LOGIT_ADJUSTMENT_TAU
        
        return LogitAdjustmentLoss(
            class_frequencies=class_frequencies,
            tau=tau,
            base_loss=config.LOGIT_ADJUSTMENT_BASE_LOSS,
            label_smoothing=label_smoothing,
            class_weights=weights_tensor
        )

    else:  # crossentropy
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
