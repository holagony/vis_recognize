import os
import glob
import numpy as np
from collections import Counter
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.metric import calculate_metrics
from models.vismfn.model import VisMFN
from models.resnet.resnet_cbam import resnet50_cbam
from datasets.vis_dataset import VisibilityDataset, InputResize
from datasets.vis_dataloader import collate_fn_filter_none, worker_init_fn
from datasets.feature_extraction import feature_extraction_block
from utils import config
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(model_path):
    '''
    加载训练好的模型
    '''
    try:
        checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=True)
    except:
        checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)

    if config.MODEL_TYPE == 'vismfn':
        if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
            model_kwargs = checkpoint['model_config'].copy()
        else:
            model_kwargs = config.get_model_kwargs()
        model = VisMFN(**model_kwargs)

    elif config.MODEL_TYPE == 'resnet':
        model = resnet50_cbam(pretrained=False, in_channels=26)

    # 加载模型权重
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'accuracy' in checkpoint:
            print(f"模型准确率: {checkpoint['accuracy']:.4f}")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(config.DEVICE)
    model.eval()

    return model


def load_test_images(data_dir_path):
    '''
    从已按类别分子文件夹的目录中加载图像路径和标签
    '''
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

        img_patterns = ["*.jpg", "*.jpeg", "*.png"]
        for pattern in img_patterns:
            for img_path in glob.glob(os.path.join(class_dir, pattern)):
                all_image_paths.append(img_path)
                all_labels.append(label_to_idx[label_str])

    return all_image_paths, all_labels


def evaluate_dataset(model, data_loader):
    '''
    评估整个数据集
    '''
    all_predictions = []
    all_labels = []
    all_probabilities = []

    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(data_loader):
            if batch_data is None:
                continue

            original_images, augmented_images, labels = batch_data
            original_images = original_images.to(config.DEVICE)
            augmented_images = augmented_images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # 特征提取：使用批处理方式
            features, num_channels = feature_extraction_block(original_images, augmented_images)
            
            # 对融合后的特征进行标准化
            # 前3个通道使用ImageNet标准化参数
            rgb_mean = [0.485, 0.456, 0.406]
            rgb_std = [0.229, 0.224, 0.225]
            
            # 其余通道使用默认参数
            other_mean = [0.0] * (num_channels - 3)
            other_std = [1.0] * (num_channels - 3)
            
            # 标准化
            mean = rgb_mean + other_mean
            std = rgb_std + other_std
            mean_tensor = torch.tensor(mean, device=features.device).view(-1, 1, 1)
            std_tensor = torch.tensor(std, device=features.device).view(-1, 1, 1)
            features = (features - mean_tensor) / std_tensor

            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def run_dataset_evaluation(model_path):
    '''
    对数据集进行完整评估
    '''
    model = load_model(model_path)  # 加载模型

    # 创建数据集和数据加载器
    test_paths, test_labels = load_test_images(config.TEST_DATA_ROOT)  # 加载测试数据
    test_dataset = VisibilityDataset(test_paths, test_labels, is_train=False, augment=False)
    test_loader = DataLoader(test_dataset, 
                             batch_size=config.BATCH_SIZE, 
                             shuffle=False, 
                             num_workers=0, 
                             pin_memory=True, 
                             collate_fn=collate_fn_filter_none, 
                             worker_init_fn=worker_init_fn)

    # 显示数据集统计信息
    label_counts = Counter(test_labels)
    for i in range(config.NUM_CLASSES):
        count = label_counts.get(i, 0)
        percentage = count / len(test_labels) * 100 if test_labels else 0
        print(f"  Level {i}: {count} 样本 ({percentage:.1f}%)")

    # 评估
    test_pred, test_true, test_prob = evaluate_dataset(model, test_loader)
    info, report = calculate_metrics(test_true, test_pred, num_classes=5)

    # 生成混淆矩阵
    cm = confusion_matrix(test_true, test_pred)
    
    # 创建混淆矩阵可视化
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Level_{i}' for i in range(config.NUM_CLASSES)],
                yticklabels=[f'Level_{i}' for i in range(config.NUM_CLASSES)])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # 保存混淆矩阵图
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return info, report


def run_single_image_inference(image_path, model_path):
    '''
    执行单张图像推理
    根据最新的dataset逻辑，需要先进行特征提取，然后标准化
    '''
    print(f"开始处理图像: {image_path}")

    # 加载模型
    model = load_model(model_path)
    print(f"模型加载成功，类型: {config.MODEL_TYPE}")

    image = Image.open(image_path).convert('RGB')
    print(f"原始图像尺寸: {image.size}")

    # 使用与dataset相同的预处理流程
    size_transform = InputResize(config.TARGET_INPUT_SIZE, direct_resize=config.DIRECT_RESIZE)
    tensor_transform = transforms.ToTensor()
    
    image = size_transform(image)
    original_image = tensor_transform(image) # 转换为tensor，范围[0,1]
    augmented_image = original_image.clone()
    
    # 将单张图像转换为批处理格式 (1, 3, H, W)
    original_batch = original_image.unsqueeze(0)  # (3, H, W) -> (1, 3, H, W)
    augmented_batch = augmented_image.unsqueeze(0)  # (3, H, W) -> (1, 3, H, W)
    
    # 特征提取：使用批处理方式
    features, num_channels = feature_extraction_block(original_batch, augmented_batch)

    # 特征标准化
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    other_mean = [0.0] * (num_channels - 3)
    other_std = [1.0] * (num_channels - 3)
    mean = rgb_mean + other_mean
    std = rgb_std + other_std
    mean_tensor = torch.tensor(mean).view(-1, 1, 1).to(features.device)
    std_tensor = torch.tensor(std).view(-1, 1, 1).to(features.device)
    features = (features - mean_tensor) / std_tensor

    # 确保特征在正确的设备上
    features = features.to(config.DEVICE)
    print(f"最终输入tensor形状: {features.shape}")

    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_label = predicted_class.item()
    all_probs = probabilities.cpu().numpy().squeeze()

    print(f"推理完成！")
    print(f"预测等级: {predicted_label}")
    print(f"置信度: {confidence.item():.4f}")
    print(f"各类别概率: {[f'{p:.4f}' for p in all_probs]}")

    # 返回详细结果
    result = {
        'image_path': image_path,
        'predicted_label': predicted_label,
        'confidence': confidence.item(),
        'probabilities': all_probs.tolist(),
        'model_type': config.MODEL_TYPE,
        'feature_channels': num_channels,
        'feature_shape': list(features.shape),
    }

    return result


if __name__ == "__main__":
    model_path = r'C:/Users/mjynj/Desktop/vis_recognize/results/models/vis_best.pth'
    info, report = run_dataset_evaluation(model_path)
    print(info)
    print('---------------------------------')
    print()
    print(report)