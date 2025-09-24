import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.metric import calculate_metrics
from models.resnet.resnet_cbam import resnet50_cbam
from models.resnet.resnet import resnet50, resnet34, resnet18, JointModel
from models.wuhan.encoder import Encoder
from models.efficientnet.efficientnet import (
    efficientnet_b0, efficientnet_b1,
    efficientnet_b0_supcon, efficientnet_b1_supcon, efficientnet_b2_supcon)
from datasets.vis_dataset import VisibilityDataset, InputResize
from datasets.vis_dataloader import collate_fn_filter_none, worker_init_fn
from datasets.feature_extraction import feature_extraction_block
from utils.utils import normalize_feature_channels
from utils import config
from tqdm import tqdm



def load_model(model_path):
    '''
    加载训练好的模型
    '''
    try:
        checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=True)
    except:
        checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)

    if config.MODEL_TYPE == 'wuhan':
        model = Encoder(3, config.NUM_CLASSES, use_dropout=False)

    elif config.MODEL_TYPE == 'resnet18':
        model = resnet18(in_channels=11, use_se=True, use_dilation=True, dilation_rates=[1, 1, 1, 2], use_psa=False)
    
    elif config.MODEL_TYPE == 'resnet34':
        model = resnet34(in_channels=11, use_se=True, use_dilation=True, dilation_rates=[1, 1, 1, 2], use_psa=False)
    
    elif config.MODEL_TYPE == 'resnet50':
        model = resnet50(in_channels=11, use_se=True, use_dilation=True, dilation_rates=[1, 1, 1, 2], use_psa=False)

    elif config.MODEL_TYPE == 'resnet18_supcon':
        base_encoder = resnet18(in_channels=11, use_se=True, use_dilation=True, dilation_rates=[1, 1, 1, 2], use_psa=False)
        model = JointModel(base_encoder, projection_dim=128, num_classes=config.NUM_CLASSES)
    
    elif config.MODEL_TYPE == 'resnet34_supcon':
        base_encoder = resnet34(in_channels=11, use_se=True, use_dilation=True, dilation_rates=[1, 1, 1, 2], use_psa=False)
        model = JointModel(base_encoder, projection_dim=128, num_classes=config.NUM_CLASSES)
    
    elif config.MODEL_TYPE == 'resnet50_supcon':
        base_encoder = resnet50(in_channels=11, use_se=True, use_dilation=True, dilation_rates=[1, 1, 1, 2], use_psa=False)
        model = JointModel(base_encoder, projection_dim=128, num_classes=config.NUM_CLASSES)
    
    # EfficientNet系列
    elif config.MODEL_TYPE == 'efficientnet_b0':
        model = efficientnet_b0(in_channels=11, num_classes=config.NUM_CLASSES, pretrained=False)
    
    elif config.MODEL_TYPE == 'efficientnet_b1':
        model = efficientnet_b1(in_channels=11, num_classes=config.NUM_CLASSES, pretrained=False)
    
    elif config.MODEL_TYPE == 'efficientnet_b0_supcon':
        model = efficientnet_b0_supcon(in_channels=11, projection_dim=128, num_classes=config.NUM_CLASSES, pretrained=False)
    
    elif config.MODEL_TYPE == 'efficientnet_b1_supcon':
        model = efficientnet_b1_supcon(in_channels=11, projection_dim=128, num_classes=config.NUM_CLASSES, pretrained=False)
    
    elif config.MODEL_TYPE == 'efficientnet_b2_supcon':
        model = efficientnet_b2_supcon(in_channels=11, projection_dim=128, num_classes=config.NUM_CLASSES, pretrained=False)
    
    else:
        raise ValueError(f"不支持的模型类型: {config.MODEL_TYPE}")

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

            if config.MODEL_TYPE == 'wuhan':
                # original_images = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(original_images)
                features = original_images
            else:
                features, num_channels = feature_extraction_block(original_images, augmented_images)
                features = normalize_feature_channels(features, depth_ch=1) # 各通道标准化

            if config.MODEL_TYPE == 'supcon':
                h, z, logits = model(features)
                outputs = logits  # 使用分类输出进行预测
            else:
                outputs = model(features)

            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def run_dataset_evaluation(model_path, data_path):
    '''
    对数据集进行完整评估，总流程
    '''
    model = load_model(model_path)
    test_paths, test_labels = load_test_images(data_path)  # 加载测试数据
    test_dataset = VisibilityDataset(test_paths, test_labels, augment=False, is_train=False, resize_mode=config.VAL_RESIZE_MODE)
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
    info, report, cm = calculate_metrics(test_true, test_pred, num_classes=5)

    # 创建混淆矩阵可视化
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                annot_kws={'size': 14},
                xticklabels=[f'Level_{i}' for i in range(config.NUM_CLASSES)],
                yticklabels=[f'Level_{i}' for i in range(config.NUM_CLASSES)])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return info, report


def run_single_image_inference(image_path, model_path):
    '''
    执行单张图像推理
    根据最新的dataset逻辑，需要先进行特征提取，然后标准化
    '''

    model = load_model(model_path)
    image = Image.open(image_path).convert('RGB')

    # 使用与dataset相同的预处理流程
    size_transform = InputResize(config.TARGET_INPUT_SIZE, is_train=False, resize_mode=config.VAL_RESIZE_MODE)
    tensor_transform = transforms.ToTensor()
    
    image = size_transform(image)
    original_image = tensor_transform(image) # 转换为tensor，范围[0,1]
    augmented_image = original_image.clone()
    
    original_batch = original_image.unsqueeze(0)  # (3, H, W) -> (1, 3, H, W)
    augmented_batch = augmented_image.unsqueeze(0)  # (3, 3, H, W) -> (1, 3, H, W)
    features, num_channels = feature_extraction_block(original_batch, augmented_batch)
    features = normalize_feature_channels(features, depth_ch=1)
    features = features.to(config.DEVICE)

    with torch.no_grad():
        # 根据模型类型进行推理
        if config.MODEL_TYPE == 'supcon':
            h, z, logits = model(features)
            outputs = logits  # 使用分类输出进行预测
        else:
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
        'feature_shape': list(features.shape)}

    return result


if __name__ == "__main__":    
    # model_path = r'C:/Users/mjynj/Desktop/Encoder_30_test_loss-0.5486_test_dice-0.8457.pt'
    # data_path = r'C:\Users\mjynj\Desktop\vis\app\data\highway_validate_data'

    model_path = r'C:/Users/mjynj/Desktop/vis_best.pth'
    data_path = config.TEST_DATA_ROOT
    
    # 检查模型类型
    print(f"当前模型类型: {config.MODEL_TYPE}")
    info, report = run_dataset_evaluation(model_path, data_path)

    print(info)
    print('---------------------------------')
    print()
    print(report)