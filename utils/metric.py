import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix


def calculate_metrics(y_true, y_pred, num_classes=5):
    '''
    计算详细的分类指标
    优化使用classification_report，避免重复计算
    '''
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_pred = np.clip(y_pred, 0, num_classes - 1)

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
            'class_ts_scores': [0.00] * num_classes,
            'mean_recall': 0.00,
            'mean_precision': 0.00,
            'mean_f1': 0.00,
            'mean_ts_score': 0.00,
            'imbalance_ratio': 1.00,
            'class_counts': [0] * num_classes
        }
        return info, None

    # 使用 classification_report 一次性计算所有指标
    report = classification_report(y_true, y_pred, labels=list(range(num_classes)), output_dict=True, zero_division=0)

    # 计算混淆矩阵（仅用于TS评分计算）
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # 从classification_report中提取各类别指标（避免重复计算）
    class_recalls = [round(report[str(i)]['recall'] * 100, 2) for i in range(num_classes)]
    class_precisions = [round(report[str(i)]['precision'] * 100, 2) for i in range(num_classes)]
    class_f1s = [round(report[str(i)]['f1-score'] * 100, 2) for i in range(num_classes)]

    # 计算每个类别的TS评分（Threat Score，IoU的另一种表示）
    class_ts_scores = []
    for i in range(num_classes):
        tp = cm[i, i]  # TP: 真正例（对角线上的值）
        fp = np.sum(cm[:, i]) - tp  # FP: 假正例（该列中除了TP之外的所有值之和）
        fn = np.sum(cm[i, :]) - tp  # FN: 假负例（该行中除了TP之外的所有值之和）
        ts_score = tp / (tp + fp + fn) if tp + fp + fn != 0 else 0.0
        class_ts_scores.append(round(ts_score * 100, 2))

    # 类别不平衡指标
    class_counts = np.bincount(y_true, minlength=num_classes)
    imbalance_ratio = round(np.max(class_counts) / (np.min(class_counts) + 1e-8), 2)

    # 构建结果字典，优先使用classification_report的结果
    info = {
        'overall_accuracy': round(report['accuracy'] * 100, 2),
        'balanced_accuracy': round(balanced_accuracy_score(y_true, y_pred) * 100, 2),
        'macro_f1': round(report['macro avg']['f1-score'] * 100, 2),
        'weighted_f1': round(report['weighted avg']['f1-score'] * 100, 2),
        'class_recalls': class_recalls,
        'class_precisions': class_precisions,
        'class_f1s': class_f1s,
        'class_ts_scores': class_ts_scores,
        'mean_recall': round(report['macro avg']['recall'] * 100, 2),  # 直接使用macro avg
        'mean_precision': round(report['macro avg']['precision'] * 100, 2),  # 直接使用macro avg
        'mean_f1': round(report['macro avg']['f1-score'] * 100, 2),  # 直接使用macro avg
        'mean_ts_score': round(np.mean(class_ts_scores), 2),
        'imbalance_ratio': imbalance_ratio,
        'class_counts': class_counts.tolist()
    }
    return info, report, cm


def calculate_class_frequencies(labels, num_classes=5):
    '''
    计算类别频率，用于logit adjustment
    Args:
        labels: 标签列表或数组
        num_classes: 类别数量
    Returns:
        dict: 包含类别频率和不平衡指标的字典
    '''
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    labels = np.asarray(labels).flatten()
    
    # 计算每个类别的数量
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    # 计算频率
    class_frequencies = class_counts / total_samples
    
    # 计算不平衡指标
    max_freq = np.max(class_frequencies)
    min_freq = np.min(class_frequencies[class_frequencies > 0])  # 排除0频率
    imbalance_ratio = max_freq / min_freq if min_freq > 0 else float('inf')
    
    # 计算logit adjustment的调整值（tau=1.0时）
    logit_adjustments = np.log(class_frequencies + 1e-12)
    
    # 计算有效样本数（Effective Number of Samples）
    # 用于Class-Balanced Loss等方法
    beta = 0.9999  # 常用值
    effective_nums = (1 - np.power(beta, class_counts)) / (1 - beta)
    cb_weights = (1 - beta) / effective_nums
    cb_weights = cb_weights / np.sum(cb_weights) * num_classes  # 归一化
    
    return {
        'class_counts': class_counts.tolist(),
        'class_frequencies': class_frequencies.tolist(),
        'total_samples': int(total_samples),
        'imbalance_ratio': round(imbalance_ratio, 4),
        'logit_adjustments': logit_adjustments.tolist(),
        'effective_numbers': effective_nums.tolist(),
        'cb_weights': cb_weights.tolist(),
        'entropy': round(-np.sum(class_frequencies * np.log(class_frequencies + 1e-12)), 4)
    }


def analyze_prediction_bias(y_true, y_pred, num_classes=5):
    '''
    分析预测偏差，特别适用于不平衡数据集
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        num_classes: 类别数量
    Returns:
        dict: 包含偏差分析的字典
    '''
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # 计算真实分布和预测分布
    true_dist = np.bincount(y_true, minlength=num_classes) / len(y_true)
    pred_dist = np.bincount(y_pred, minlength=num_classes) / len(y_pred)
    
    # 计算KL散度
    kl_div = np.sum(true_dist * np.log((true_dist + 1e-12) / (pred_dist + 1e-12)))
    
    # 计算每个类别的预测偏差
    bias_per_class = pred_dist - true_dist
    
    # 计算总变差距离（Total Variation Distance）
    tv_distance = 0.5 * np.sum(np.abs(bias_per_class))
    
    return {
        'true_distribution': true_dist.tolist(),
        'predicted_distribution': pred_dist.tolist(),
        'bias_per_class': bias_per_class.tolist(),
        'kl_divergence': round(kl_div, 6),
        'tv_distance': round(tv_distance, 6),
        'max_bias_class': int(np.argmax(np.abs(bias_per_class))),
        'max_bias_value': round(float(np.max(np.abs(bias_per_class))), 6)
    }
