import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix


def calculate_metrics(y_true, y_pred, num_classes=5):
    """
    计算详细的分类指标，特别适合不平衡数据集
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签  
        num_classes: 类别数量
    
    Returns:
        dict: 包含各种指标的字典
    """
    # 转换为numpy数组
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # 确保是1D数组
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # 确保预测标签在合法范围内，避免sklearn警告
    y_pred = np.clip(y_pred, 0, num_classes - 1)
    
    # 总体准确率
    overall_acc = np.mean(y_true == y_pred) * 100
    
    # 平衡准确率（各类别召回率的平均值）
    # 使用labels参数指定所有可能的类别，避免警告
    balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100
    
    # 计算F1分数（宏平均和加权平均）
    from sklearn.metrics import f1_score
    # 指定labels参数避免警告
    labels = list(range(num_classes))
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0) * 100
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0) * 100
    
    # 如果没有预测结果，返回默认值
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'overall_accuracy': 0.0,
            'balanced_accuracy': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'class_recalls': [0.0] * num_classes,
            'class_precisions': [0.0] * num_classes,
            'class_f1s': [0.0] * num_classes,
            'mean_recall': 0.0,
            'mean_precision': 0.0,
            'mean_f1': 0.0,
            'imbalance_ratio': 1.0,
            'class_counts': [0] * num_classes
        }
    
    # 各类别指标
    class_accuracies = []
    class_recalls = []
    class_precisions = []
    class_f1s = []
    
    for i in range(num_classes):
        # 类别i的样本
        class_mask = (y_true == i)
        class_mask_sum = np.sum(class_mask) if hasattr(class_mask, 'sum') else int(class_mask)
        
        if class_mask_sum > 0:  # 如果该类别有样本
            class_acc = np.mean(y_pred[class_mask] == i) * 100  # 召回率
            class_recalls.append(class_acc)
        else:
            class_recalls.append(0.0)
        
        # 类别i的精确率
        pred_mask = (y_pred == i)
        pred_mask_sum = np.sum(pred_mask) if hasattr(pred_mask, 'sum') else int(pred_mask)
        
        if pred_mask_sum > 0:
            class_prec = np.mean(y_true[pred_mask] == i) * 100
            class_precisions.append(class_prec)
        else:
            class_precisions.append(0.0)
        
        # 类别i的F1分数
        if class_mask_sum > 0 or pred_mask_sum > 0:
            # 使用二分类F1计算，避免sklearn警告
            class_f1 = f1_score(y_true == i, y_pred == i, zero_division=0) * 100
            class_f1s.append(class_f1)
        else:
            class_f1s.append(0.0)
    
    # 计算类别不平衡指标
    class_counts = np.bincount(y_true, minlength=num_classes)
    imbalance_ratio = np.max(class_counts) / (np.min(class_counts) + 1e-8)  # 避免除零
    
    return {
        'overall_accuracy': overall_acc,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'class_recalls': class_recalls,
        'class_precisions': class_precisions,
        'class_f1s': class_f1s,
        'mean_recall': np.mean(class_recalls),
        'mean_precision': np.mean(class_precisions),
        'mean_f1': np.mean(class_f1s),
        'imbalance_ratio': imbalance_ratio,
        'class_counts': class_counts.tolist()
    }
