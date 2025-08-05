# inference.py - 高速公路能见度评估模型推理脚本
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import argparse
import logging
from datetime import datetime
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                             balanced_accuracy_score, precision_recall_fscore_support,
                             cohen_kappa_score)
from torch.utils.data import DataLoader
import pandas as pd

from model.vis_mfn import VisMFN  
from dataset import VisibilityDataset, InputResize, collate_fn_filter_none
from utils import config

def setup_logging(output_dir):
    """设置简化的日志记录"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def calculate_detailed_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    计算详细的不平衡多分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率 (可选)
        class_names: 类别名称 (可选)
    
    Returns:
        dict: 包含各种指标的字典
    """
    if class_names is None:
        class_names = [f'Level {i}' for i in range(config.NUM_CLASSES)]
    
    # 基础指标
    overall_acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # 各类别指标
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)), zero_division=0
    )
    
    # 计算各类别准确率
    class_accuracies = []
    for i in range(config.NUM_CLASSES):
        class_mask = (y_true == i)
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    
    # 计算每个类别的样本分布
    unique, counts = np.unique(y_true, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    
    metrics = {
        'overall_accuracy': overall_acc,
        'balanced_accuracy': balanced_acc,
        'cohen_kappa': kappa,
        'macro_precision': np.mean(precision),
        'macro_recall': np.mean(recall),
        'macro_f1': np.mean(f1),
        'weighted_precision': np.average(precision, weights=support),
        'weighted_recall': np.average(recall, weights=support),
        'weighted_f1': np.average(f1, weights=support),
        'class_precision': precision,
        'class_recall': recall,
        'class_f1': f1,
        'class_accuracies': np.array(class_accuracies),
        'class_support': support,
        'class_distribution': class_distribution,
        'confusion_matrix': cm,
        'class_names': class_names
    }
    
    return metrics


def find_best_model(model_dir):
    """在模型目录中查找最佳模型文件"""
    possible_names = [
        "vis_mfn_best.pth",
        "best_model.pth",
        "vis_mfn_final.pth",
        "final_model.pth"
    ]
    
    for name in possible_names:
        model_path = os.path.join(model_dir, name)
        if os.path.exists(model_path):
            return model_path
    
    # 查找epoch文件
    if os.path.exists(model_dir):
        epoch_files = [f for f in os.listdir(model_dir) 
                      if f.startswith("vis_mfn_epoch_") and f.endswith(".pth")]
        if epoch_files:
            epoch_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            return os.path.join(model_dir, epoch_files[-1])
    
    return None

def load_data_from_predefined_dir(data_dir_path):
    """从已按类别分子文件夹的目录中加载图像路径和标签"""
    all_image_paths = []
    all_labels = []
    expected_labels_str = [str(i) for i in range(config.NUM_CLASSES)]
    label_to_idx = {label_str: idx for idx, label_str in enumerate(expected_labels_str)}

    if not os.path.isdir(data_dir_path):
        return [], []

    for label_str in expected_labels_str:
        class_dir = os.path.join(data_dir_path, label_str)
        if not os.path.isdir(class_dir):
            continue
        
        img_patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
        for pattern in img_patterns:
            for img_path in glob.glob(os.path.join(class_dir, pattern)):
                all_image_paths.append(img_path)
                all_labels.append(label_to_idx[label_str])
    
    return all_image_paths, all_labels

def load_model(model_path, logger):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None

    try:
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        logger.info(f"成功加载检查点: {model_path}")
    except Exception as e:
        logger.error(f"加载检查点失败: {e}")
        return None

    # 获取模型配置
    try:
        if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
            model_kwargs = checkpoint['model_config'].copy()
            logger.info("使用检查点中的模型配置")
        else:
            model_kwargs = config.get_model_kwargs()
            logger.info("使用默认模型配置")
        
        model = VisMFN(**model_kwargs)
        
        # 加载模型权重
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'accuracy' in checkpoint:
                logger.info(f"模型准确率: {checkpoint['accuracy']:.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        logger.info("模型加载成功")
        model.eval()
        return model
        
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        return None

def evaluate_dataset(model, data_loader, logger):
    """评估整个数据集"""
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            if inputs is None or labels is None:
                continue
            
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    logger.info(f"评估完成，共处理 {len(all_predictions)} 个样本")
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def create_evaluation_visualizations(metrics, dataset_name, output_dir):
    """创建详细的评估可视化图表"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建一个包含多个子图的大图
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 混淆矩阵
    plt.subplot(2, 3, 1)
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=metrics['class_names'],
                yticklabels=metrics['class_names'])
    plt.title(f'混淆矩阵\n总体准确率: {metrics["overall_accuracy"]:.3f}')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    # 2. 各类别召回率
    plt.subplot(2, 3, 2)
    bars = plt.bar(range(config.NUM_CLASSES), metrics['class_recall'])
    plt.title(f'各类别召回率\n平衡准确率: {metrics["balanced_accuracy"]:.3f}')
    plt.xlabel('能见度等级')
    plt.ylabel('召回率')
    plt.xticks(range(config.NUM_CLASSES))
    plt.ylim(0, 1)
    for i, v in enumerate(metrics['class_recall']):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 3. 各类别精确率
    plt.subplot(2, 3, 3)
    bars = plt.bar(range(config.NUM_CLASSES), metrics['class_precision'])
    plt.title(f'各类别精确率\nKappa系数: {metrics["cohen_kappa"]:.3f}')
    plt.xlabel('能见度等级')
    plt.ylabel('精确率')
    plt.xticks(range(config.NUM_CLASSES))
    plt.ylim(0, 1)
    for i, v in enumerate(metrics['class_precision']):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 4. 各类别F1分数
    plt.subplot(2, 3, 4)
    bars = plt.bar(range(config.NUM_CLASSES), metrics['class_f1'])
    plt.title(f'各类别F1分数\n宏平均F1: {metrics["macro_f1"]:.3f}')
    plt.xlabel('能见度等级')
    plt.ylabel('F1分数')
    plt.xticks(range(config.NUM_CLASSES))
    plt.ylim(0, 1)
    for i, v in enumerate(metrics['class_f1']):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 5. 样本分布
    plt.subplot(2, 3, 5)
    class_counts = [metrics['class_distribution'].get(i, 0) for i in range(config.NUM_CLASSES)]
    bars = plt.bar(range(config.NUM_CLASSES), class_counts)
    plt.title('各类别样本分布')
    plt.xlabel('能见度等级')
    plt.ylabel('样本数量')
    plt.xticks(range(config.NUM_CLASSES))
    for i, v in enumerate(class_counts):
        plt.text(i, v + max(class_counts)*0.01, str(v), ha='center', va='bottom')
    
    # 6. 综合指标对比
    plt.subplot(2, 3, 6)
    metric_names = ['总体准确率', '平衡准确率', '宏平均精确率', '宏平均召回率', '宏平均F1']
    metric_values = [
        metrics['overall_accuracy'], 
        metrics['balanced_accuracy'],
        metrics['macro_precision'],
        metrics['macro_recall'], 
        metrics['macro_f1']
    ]
    bars = plt.bar(range(len(metric_names)), metric_values)
    plt.title('综合指标对比')
    plt.xticks(range(len(metric_names)), metric_names, rotation=45, ha='right')
    plt.ylabel('分数')
    plt.ylim(0, 1)
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    chart_file = os.path.join(output_dir, f"evaluation_report_{dataset_name}_{timestamp}.png")
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_file


def save_detailed_results(y_true, y_pred, y_prob, dataset_name, output_dir, logger):
    """保存详细的评估结果"""
    if len(y_true) == 0:
        logger.warning(f"无数据可保存: {dataset_name}")
        return {}
    
    # 计算详细指标
    metrics = calculate_detailed_metrics(y_true, y_pred, y_prob)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建可视化图表
    chart_file = create_evaluation_visualizations(metrics, dataset_name, output_dir)
    logger.info(f"评估图表已保存: {chart_file}")
    
    # 保存详细报告
    report_file = os.path.join(output_dir, f"detailed_report_{dataset_name}_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"=== {dataset_name.upper()} 详细评估报告 ===\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总样本数: {len(y_true)}\n\n")
        
        f.write("=== 整体指标 ===\n")
        f.write(f"总体准确率: {metrics['overall_accuracy']:.4f}\n")
        f.write(f"平衡准确率: {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n\n")
        
        f.write("=== 宏平均指标 ===\n")
        f.write(f"宏平均精确率: {metrics['macro_precision']:.4f}\n")
        f.write(f"宏平均召回率: {metrics['macro_recall']:.4f}\n")
        f.write(f"宏平均F1分数: {metrics['macro_f1']:.4f}\n\n")
        
        f.write("=== 加权平均指标 ===\n")
        f.write(f"加权平均精确率: {metrics['weighted_precision']:.4f}\n")
        f.write(f"加权平均召回率: {metrics['weighted_recall']:.4f}\n")
        f.write(f"加权平均F1分数: {metrics['weighted_f1']:.4f}\n\n")
        
        f.write("=== 各类别详细指标 ===\n")
        f.write(f"{'类别':<8} {'样本数':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'准确率':<8}\n")
        f.write("-" * 60 + "\n")
        for i in range(config.NUM_CLASSES):
            support = metrics['class_support'][i]
            precision = metrics['class_precision'][i]
            recall = metrics['class_recall'][i]
            f1 = metrics['class_f1'][i]
            accuracy = metrics['class_accuracies'][i]
            f.write(f"Level {i:<3} {support:<8} {precision:<8.3f} {recall:<8.3f} {f1:<8.3f} {accuracy:<8.3f}\n")
        
        f.write(f"\n=== 混淆矩阵 ===\n")
        f.write("预测 →  ")
        for i in range(config.NUM_CLASSES):
            f.write(f"L{i:<5}")
        f.write("\n")
        for i in range(config.NUM_CLASSES):
            f.write(f"L{i} ↓    ")
            for j in range(config.NUM_CLASSES):
                f.write(f"{metrics['confusion_matrix'][i][j]:<6}")
            f.write("\n")
    
    logger.info(f"详细报告已保存: {report_file}")
    
    # 保存预测结果CSV
    results_data = {
        'true_label': y_true, 
        'predicted_label': y_pred, 
        'correct': y_true == y_pred
    }
    
    # 添加概率信息
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == config.NUM_CLASSES:
        for i in range(config.NUM_CLASSES):
            results_data[f'prob_level_{i}'] = y_prob[:, i]
    
    results_df = pd.DataFrame(results_data)
    csv_file = os.path.join(output_dir, f"predictions_{dataset_name}_{timestamp}.csv")
    results_df.to_csv(csv_file, index=False)
    logger.info(f"预测结果已保存: {csv_file}")
    
    # 输出关键指标到日志
    logger.info(f"\n=== {dataset_name.upper()} 评估结果 ===")
    logger.info(f"总体准确率: {metrics['overall_accuracy']:.4f}")
    logger.info(f"平衡准确率: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    logger.info(f"各类别召回率: {[f'{r:.3f}' for r in metrics['class_recall']]}")
    logger.info(f"各类别精确率: {[f'{p:.3f}' for p in metrics['class_precision']]}")
    logger.info(f"各类别F1分数: {[f'{f:.3f}' for f in metrics['class_f1']]}")
    
    return metrics

def run_dataset_evaluation(model_path, output_dir, logger):
    """对数据集进行完整评估"""
    model = load_model(model_path, logger)
    if model is None:
        return False

    # 加载数据
    val_paths, val_labels = load_data_from_predefined_dir(config.VAL_DATA_ROOT)
    if not val_paths:
        logger.error("未找到验证数据")
        return False

    # 显示数据集统计信息
    from collections import Counter
    label_counts = Counter(val_labels)
    logger.info(f"验证集统计信息:")
    for i in range(config.NUM_CLASSES):
        count = label_counts.get(i, 0)
        percentage = count / len(val_labels) * 100 if val_labels else 0
        logger.info(f"  Level {i}: {count} 样本 ({percentage:.1f}%)")

    # 创建数据集和数据加载器
    val_dataset = VisibilityDataset(val_paths, val_labels, is_train=False, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True, collate_fn=collate_fn_filter_none)
    
    logger.info(f"开始评估 {len(val_paths)} 张图片")
    
    # 评估
    val_pred, val_true, val_prob = evaluate_dataset(model, val_loader, logger)
    
    if len(val_pred) > 0:
        # 使用新的详细评估函数
        metrics = save_detailed_results(val_true, val_pred, val_prob, "validation_set", output_dir, logger)
        return True
    else:
        logger.error("评估失败，无有效数据")
        return False

def run_single_image_inference(image_path, model_path, output_dir, logger):
    """执行单张图像推理"""
    os.makedirs(output_dir, exist_ok=True)
    
    model = load_model(model_path, logger)
    if model is None:
        return False

    # 预处理图像
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            InputResize(config.TARGET_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
        ])
        input_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    except Exception as e:
        logger.error(f"图像预处理失败: {e}")
        return False

    # 推理
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
    predicted_label = predicted_class.item()
    all_probs = probabilities.cpu().numpy().squeeze()
    
    logger.info(f"推理结果: 预测等级 {predicted_label}, 置信度 {confidence.item():.4f}")
    
    # 可视化结果
    try:
        plt.figure(figsize=(12, 6))
        
        # 显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"输入图像\n{os.path.basename(image_path)}")
        plt.axis('off')
        
        # 显示预测结果
        plt.subplot(1, 2, 2)
        bars = plt.bar(range(config.NUM_CLASSES), all_probs)
        plt.title(f"预测结果\n等级: {predicted_label} (置信度: {confidence.item():.3f})")
        plt.xlabel("能见度等级")
        plt.ylabel("概率")
        plt.xticks(range(config.NUM_CLASSES))
        bars[predicted_label].set_color('red')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_result_{timestamp}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"结果已保存到: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"可视化失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Highway Visibility Estimation Inference')
    parser.add_argument('--mode', choices=['single', 'dataset'], default='dataset', 
                       help='推理模式: single=单张图像, dataset=整个数据集')
    parser.add_argument('--image', '-i', type=str, help='单张图像推理时的输入图像路径')
    parser.add_argument('--model', '-m', type=str, help='模型文件路径')
    parser.add_argument('--output', '-o', type=str, default=config.INFERENCE_RESULT_DIR, 
                       help=f'输出目录 (默认: {config.INFERENCE_RESULT_DIR})')
    parser.add_argument('--model_dir', type=str, help='模型目录路径 (用于自动查找最佳模型)')
    
    args = parser.parse_args()
    
    logger = setup_logging(args.output)
    logger.info(f"开始推理 - 模式: {args.mode}")
    
    # 确定模型路径
    model_path = args.model
    if not model_path and args.model_dir:
        model_path = find_best_model(args.model_dir)
        if model_path:
            logger.info(f"自动找到模型: {model_path}")
    
    if not model_path or not os.path.exists(model_path):
        logger.error("未找到有效的模型文件")
        return
    
    logger.info(f"使用模型: {model_path}")
    logger.info(f"输出目录: {args.output}")
    
    if args.mode == 'single':
        if not args.image:
            logger.error("单张图像模式需要指定 --image 参数")
            return
        if not os.path.exists(args.image):
            logger.error(f"输入图像不存在: {args.image}")
            return
        
        success = run_single_image_inference(args.image, model_path, args.output, logger)
        logger.info(f"单张图像推理 {'成功' if success else '失败'}")
        
    elif args.mode == 'dataset':
        success = run_dataset_evaluation(model_path, args.output, logger)
        logger.info(f"数据集评估 {'成功' if success else '失败'}")

if __name__ == '__main__':
    main() 