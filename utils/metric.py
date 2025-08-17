import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report


def calculate_metrics(y_true, y_pred, num_classes=5):
    '''
    计算详细的分类指标，适合不平衡数据集
    使用 classification_report 简化代码
    '''
    # 转换为numpy数组
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_pred = np.clip(y_pred, 0, num_classes - 1)  # 确保预测标签在合法范围内
    
    # 如果没有预测结果，返回默认值
    if len(y_true) == 0 or len(y_pred) == 0:
        info = {
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
            'class_counts': [0] * num_classes}
        return info, None
    
    # 使用 classification_report 计算各类别指标
    report = classification_report(y_true, y_pred, labels=list(range(num_classes)), output_dict=True, zero_division=0)
    
    # 提取各类别的 precision, recall, f1
    class_recalls = []
    class_precisions = []
    class_f1s = []
    
    for i in range(num_classes):
        class_recalls.append(report[str(i)]['recall'] * 100)
        class_precisions.append(report[str(i)]['precision'] * 100)
        class_f1s.append(report[str(i)]['f1-score'] * 100)
    
    # 类别不平衡指标
    class_counts = np.bincount(y_true, minlength=num_classes)
    imbalance_ratio = np.max(class_counts) / (np.min(class_counts) + 1e-8)  # 避免除零
    
    info = {
        'overall_accuracy': report['accuracy'] * 100,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred) * 100,
        'macro_f1': report['macro avg']['f1-score'] * 100,
        'weighted_f1': report['weighted avg']['f1-score'] * 100,
        'class_recalls': class_recalls,
        'class_precisions': class_precisions,
        'class_f1s': class_f1s,
        'mean_recall': report['macro avg']['recall'] * 100,
        'mean_precision': report['macro avg']['precision'] * 100,
        'mean_f1': report['macro avg']['f1-score'] * 100,
        'imbalance_ratio': imbalance_ratio,
        'class_counts': class_counts.tolist()}

    return info, report