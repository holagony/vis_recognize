import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix


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
            'overall_accuracy': 0.00,
            'balanced_accuracy': 0.00,
            'macro_f1': 0.00,
            'weighted_f1': 0.00,
            'class_recalls': [0.00] * num_classes,
            'class_precisions': [0.00] * num_classes,
            'class_f1s': [0.00] * num_classes,
            'class_ts_scores': [0.00] * num_classes,  # 新增：各类别TS评分
            'mean_recall': 0.00,
            'mean_precision': 0.00,
            'mean_f1': 0.00,
            'mean_ts_score': 0.00,  # 新增：平均TS评分
            'imbalance_ratio': 1.00,
            'class_counts': [0] * num_classes}
        return info, None
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # 计算每个类别的TS评分
    class_ts_scores = []
    for i in range(num_classes):
        # TP: 真正例（对角线上的值）
        tp = cm[i, i]
        # FP: 假正例（该列中除了TP之外的所有值之和）
        fp = np.sum(cm[:, i]) - tp
        # FN: 假负例（该行中除了TP之外的所有值之和）
        fn = np.sum(cm[i, :]) - tp
        
        # TS = TP / (TP + FP + FN)
        if tp + fp + fn == 0:
            ts_score = 0.0
        else:
            ts_score = tp / (tp + fp + fn)
        
        class_ts_scores.append(round(ts_score * 100, 2))  # 转换为百分比并保留2位小数
    
    # 计算平均TS评分
    mean_ts_score = round(np.mean(class_ts_scores), 2)
    
    # 使用 classification_report 计算各类别指标
    report = classification_report(y_true, y_pred, labels=list(range(num_classes)), output_dict=True, zero_division=0)
    
    # 提取各类别的 precision, recall, f1
    class_recalls = []
    class_precisions = []
    class_f1s = []
    
    for i in range(num_classes):
        class_recalls.append(round(report[str(i)]['recall'] * 100, 2))
        class_precisions.append(round(report[str(i)]['precision'] * 100, 2))
        class_f1s.append(round(report[str(i)]['f1-score'] * 100, 2))
    
    # 类别不平衡指标
    class_counts = np.bincount(y_true, minlength=num_classes)
    imbalance_ratio = round(np.max(class_counts) / (np.min(class_counts) + 1e-8), 2)  # 避免除零并保留2位小数
    
    info = {
        'overall_accuracy': round(report['accuracy'] * 100, 2),
        'balanced_accuracy': round(balanced_accuracy_score(y_true, y_pred) * 100, 2),
        'macro_f1': round(report['macro avg']['f1-score'] * 100, 2),
        'weighted_f1': round(report['weighted avg']['f1-score'] * 100, 2),
        'class_recalls': class_recalls,
        'class_precisions': class_precisions,
        'class_f1s': class_f1s,
        'class_ts_scores': class_ts_scores,  # 新增：各类别TS评分
        'mean_recall': round(report['macro avg']['recall'] * 100, 2),
        'mean_precision': round(report['macro avg']['precision'] * 100, 2),
        'mean_f1': round(report['macro avg']['f1-score'] * 100, 2),
        'mean_ts_score': mean_ts_score,  # 新增：平均TS评分
        'imbalance_ratio': imbalance_ratio,
        'class_counts': class_counts.tolist()}

    return info, report