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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
import pandas as pd

from model.vis_mfn import VisMFN  
from dataset import HighwayVisibilityDataset, FlexibleInputResize, collate_fn_filter_none
from utils.config import (
    DEVICE, NORM_MEAN, NORM_STD, NUM_CLASSES,
    SCENE_DEPTH_BRANCH_WEIGHTS, BATCH_SIZE,
    TARGET_INPUT_SIZE, get_model_kwargs,
    TRAIN_DATA_ROOT, VAL_DATA_ROOT, INFERENCE_RESULT_DIR
)

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
    expected_labels_str = [str(i) for i in range(NUM_CLASSES)]
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
        checkpoint = torch.load(model_path, map_location=DEVICE)
        logger.info(f"成功加载检查点: {model_path}")
    except Exception as e:
        logger.error(f"加载检查点失败: {e}")
        return None

    # 获取模型配置
    try:
        if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
            model_kwargs = checkpoint['model_config'].copy()
            model_kwargs.pop('model_type', None)
            model_kwargs['scene_depth_weights_path'] = SCENE_DEPTH_BRANCH_WEIGHTS
            logger.info("使用检查点中的模型配置")
        else:
            model_kwargs = get_model_kwargs()
            model_kwargs['scene_depth_weights_path'] = SCENE_DEPTH_BRANCH_WEIGHTS
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
            
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    logger.info(f"评估完成，共处理 {len(all_predictions)} 个样本")
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def save_results(y_true, y_pred, y_prob, dataset_name, output_dir, logger):
    """保存评估结果"""
    if len(y_true) == 0:
        logger.warning(f"无数据可保存: {dataset_name}")
        return {}
    
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    
    # 分类报告
    report = classification_report(y_true, y_pred,
                                 target_names=[f'Level {i}' for i in range(NUM_CLASSES)],
                                 output_dict=True, zero_division=0)
    
    # 保存混淆矩阵图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Level {i}' for i in range(NUM_CLASSES)],
                yticklabels=[f'Level {i}' for i in range(NUM_CLASSES)])
    plt.title(f'{dataset_name} Confusion Matrix\nAccuracy: {accuracy:.4f}')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_file = os.path.join(output_dir, f"confusion_matrix_{dataset_name}_{timestamp}.png")
    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存预测结果
    results_data = {'true_label': y_true, 'predicted_label': y_pred, 'correct': y_true == y_pred}
    
    # 添加概率信息
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == NUM_CLASSES:
        for i in range(NUM_CLASSES):
            results_data[f'prob_level_{i}'] = y_prob[:, i]
    
    results_df = pd.DataFrame(results_data)
    csv_file = os.path.join(output_dir, f"predictions_{dataset_name}_{timestamp}.csv")
    results_df.to_csv(csv_file, index=False)
    
    logger.info(f"{dataset_name} 结果已保存 - 准确率: {accuracy:.4f}")
    
    return {'accuracy': accuracy, 'confusion_matrix': cm, 'report': report}

def run_dataset_evaluation(model_path, output_dir, logger):
    """对数据集进行完整评估"""
    model = load_model(model_path, logger)
    if model is None:
        return False

    # 加载数据
    val_paths, val_labels = load_data_from_predefined_dir(VAL_DATA_ROOT)
    if not val_paths:
        logger.error("未找到验证数据")
        return False

    # 创建数据集和数据加载器
    val_dataset = HighwayVisibilityDataset(val_paths, val_labels, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn_filter_none)
    
    logger.info(f"开始评估 {len(val_paths)} 张图片")
    
    # 评估
    val_pred, val_true, val_prob = evaluate_dataset(model, val_loader, logger)
    
    if len(val_pred) > 0:
        results = save_results(val_true, val_pred, val_prob, "test_set", output_dir, logger)
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
            FlexibleInputResize(TARGET_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
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
        bars = plt.bar(range(NUM_CLASSES), all_probs)
        plt.title(f"预测结果\n等级: {predicted_label} (置信度: {confidence.item():.3f})")
        plt.xlabel("能见度等级")
        plt.ylabel("概率")
        plt.xticks(range(NUM_CLASSES))
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
    parser.add_argument('--output', '-o', type=str, default=INFERENCE_RESULT_DIR, 
                       help=f'输出目录 (默认: {INFERENCE_RESULT_DIR})')
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