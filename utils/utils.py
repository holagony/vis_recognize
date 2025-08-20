import os
import random
import logging
import psutil
import numpy as np
import torch
import torchvision.transforms as T
from datetime import datetime


def set_seed(seed=42):
    '''''
    设置全局随机种子以确保训练的可重复性
    
    Args:
        seed (int): 随机种子值，默认为42
    '''
    # Python内置随机数
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def get_memory_usage():
    '''
    获取当前内存使用情况
    '''
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    gpu_memory = "N/A"
    if torch.cuda.is_available():
        gpu_memory = f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"

    return {'ram': f"{memory_info.rss / 1024**3:.2f}GB", 'gpu': gpu_memory}


def setup_logging(output_dir):
    '''
    设置日志记录
    返回配置好的 logging 模块，直接使用 logging.info() 等函数
    '''
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()], 
                        force=True)
    return logging


def normalize_feature_26channels(batch_features):
    '''
    对特征进行标准化 26通道
    '''
    # 为每种类型的数据设置合适的标准化参数
    imagenet_mean = [0.485, 0.456, 0.406]  # 用于RGB通道
    imagenet_std = [0.229, 0.224, 0.225]   # 用于RGB通道

    # 根据通道类型分别设置参数
    extended_mean = []
    extended_std = []

    # RGB通道 (3个通道)
    extended_mean.extend(imagenet_mean)
    extended_std.extend(imagenet_std)

    # 深度通道 (16个通道) - 范围[-1,1]，适合使用mean=0, std=1
    extended_mean.extend([0.0] * 16)
    extended_std.extend([1.0] * 16)

    # 传输通道 (1个通道) - 范围[0,1]，适合使用mean=0.5, std=0.5
    extended_mean.append(0.5)
    extended_std.append(0.5)

    # 光谱通道 (3个通道) - L[0,1], A[-0.5,0.5], B[-0.5,0.5]
    # L通道使用mean=0.5, std=0.5
    # A和B通道使用mean=0, std=0.5
    extended_mean.extend([0.5, 0.0, 0.0])
    extended_std.extend([0.5, 0.5, 0.5])

    # 细节通道 (3个通道) - 范围[-1,1]，适合使用mean=0, std=1
    extended_mean.extend([0.0] * 3)
    extended_std.extend([1.0] * 3)

    # 创建标准化transform并应用
    normalize = T.Normalize(mean=extended_mean, std=extended_std)
    batch_features = normalize(batch_features)

    return batch_features