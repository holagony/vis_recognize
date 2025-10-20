import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import uuid
import ssl
import shutil
import urllib
import urllib.request
import torch
from torch.utils.data import DataLoader
from models.resnet.resnet import resnet50, resnet34, resnet18, JointModel
from models.wuhan.encoder import Encoder
from models.efficientnet.efficientnet import (efficientnet_b0, efficientnet_b1, efficientnet_b0_supcon, efficientnet_b1_supcon, efficientnet_b2_supcon)
from datasets.vis_dataset import VisibilityDataset, InputResize
from datasets.vis_dataloader import collate_fn_filter_none, worker_init_fn
from datasets.feature_extraction import feature_extraction_block
from utils.utils import normalize_feature_channels
from utils import config


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


def map_probability_to_visibility(predicted_classes, confidences):
    visibility_ranges = [(3000, 8000), (250, 500), (150, 200), (80, 100), (20, 50)]

    return [visibility_ranges[idx][0] + (visibility_ranges[idx][1] - visibility_ranges[idx][0]) * confidences[i] for i, idx in enumerate(predicted_classes)]


def run_batch_image_inference(image_paths, model_path):
    '''
    图像推理，支持批量
    根据最新的dataset逻辑，需要先进行特征提取，然后标准化
    '''
    model = load_model(model_path)
    dummy_labels = [0] * len(image_paths)  # 创建虚拟标签（推理时不需要真实标签）

    # 使用VisibilityDataset和DataLoader
    dataset = VisibilityDataset(image_paths, dummy_labels, augment=False, is_train=False, resize_mode=config.VAL_RESIZE_MODE)
    dataloader = DataLoader(dataset, batch_size=len(image_paths), shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn_filter_none, worker_init_fn=worker_init_fn)

    # 获取批次数据
    batch_data = next(iter(dataloader))
    original_images, augmented_images, labels = batch_data

    # 进行特征提取
    if config.MODEL_TYPE == 'wuhan':
        features = original_images
    else:
        features, num_channels = feature_extraction_block(original_images, augmented_images)
        features = normalize_feature_channels(features, depth_ch=1)

    batch_tensor = features.to(config.DEVICE)

    with torch.no_grad():
        if 'supcon' in config.MODEL_TYPE:
            h, z, logits = model(batch_tensor)
            outputs = logits
        else:
            outputs = model(batch_tensor)

        probabilities = torch.softmax(outputs, dim=1)
        confidences, predicted_classes = torch.max(probabilities, 1)

    # 生成能见度数值
    predicted_classes_np = predicted_classes.cpu().numpy()
    confidences_np = confidences.cpu().numpy()
    visibility_values = map_probability_to_visibility(predicted_classes_np, confidences_np)

    results = []
    for i, image_path in enumerate(image_paths):
        filename = os.path.basename(image_path)
        result = {
            'condition': 'RV',
            'filename': filename,
            'level': int(predicted_classes[i].item()),
            'value': round(float(visibility_values[i]), 1),
            'confidence': round(float(confidences[i].item()), 3)
        }
        results.append(result)

    return results


def vis_inference(data_json):
    '''
    能见度等级推理
    '''
    uuid4 = uuid.uuid4().hex
    data_dir = os.path.join(config.IN_DATA_DIR, uuid4)  # 容器内路径
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.chmod(data_dir, 0o007 | 0o070 | 0o700)

    imgPaths = data_json['imgPaths']  # list
    if isinstance(imgPaths, str):
        imgPaths = [imgPaths]

    total_path = []
    for path in imgPaths:
        if 'http' in path:
            imgname = os.path.basename(path)  # 使用os.path.basename获取文件名，兼容所有操作系统
            save_path = os.path.join(data_dir, imgname)

            # 禁用SSL证书验证
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # 创建opener并安装
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(path, save_path)
        else:
            save_path = path.replace(config.OUT_DATA_DIR, config.IN_DATA_DIR)

        total_path.append(save_path)

    # 推理
    model_path = './model_hub/vis_best.pth'
    vis_result = run_batch_image_inference(total_path, model_path)
    result_dict = dict()
    result_dict['uuid'] = uuid4
    result_dict['vis_result'] = vis_result

    return result_dict


if __name__ == "__main__":
    import time
    import simplejson
    t1 = time.time()
    data_json = dict()
    # test_path = r'C:\Users\mjynj\Desktop\test'
    # test_paths = glob.glob(os.path.join(test_path, '*.jpg'))
    # data_json['imgPaths'] = test_paths
    data_json['imgPaths'] = ['https://www.jiazhao.com/images/Articles/month_1411/201411261020332530.png']

    result_dict = vis_inference(data_json)
    return_data = simplejson.dumps({'code': 200, 'msg': 'success', 'data': result_dict}, ensure_ascii=False, ignore_nan=True)
    print(result_dict)
    t2 = time.time()
    print(t2 - t1)
