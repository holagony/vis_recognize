import os
import random
import logging
import psutil
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from datetime import datetime
from utils import config


def set_seed(seed=42):
    '''
    固定全局随机种子
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

    gpu_memory = 'N/A' if not torch.cuda.is_available() else f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"

    return {'ram': f'{memory_info.rss / 1024**3:.2f}GB', 'gpu': gpu_memory}


def setup_logging(output_dir):
    '''
    设置日志记录
    返回配置好的 logging 模块，直接使用 logging.info() 等函数
    '''
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()], force=True)

    return logging


def normalize_feature_channels(batch_features, depth_ch=1):
    '''
    对特征进行标准化，目前11通道
    顺序：rgb_feat, depth_feat, transmission_feat, spectral_feat, detail_feat
    所有通道数值范围都在[0,1]之间，从头训练使用统一标准化
    '''
    # 根据通道类型分别设置参数
    extended_mean = []
    extended_std = []

    # RGB通道 (3个通道) - 原始图像范围[0,1]，使用mean=0.5, std=0.5
    extended_mean.extend([0.5, 0.5, 0.5])
    extended_std.extend([0.5, 0.5, 0.5])

    # 深度通道 (depth_ch个通道) - 范围[0,1]
    extended_mean.extend([0.5] * depth_ch)
    extended_std.extend([0.5] * depth_ch)

    # 传输通道 (1个通道) - 范围[0,1]
    extended_mean.append(0.5)
    extended_std.append(0.5)

    # 光谱通道 (3个通道) - 范围[0,1]
    extended_mean.extend([0.5, 0.5, 0.5])
    extended_std.extend([0.5, 0.5, 0.5])

    # 细节通道 (3个通道) - 范围[0,1]
    extended_mean.extend([0.5, 0.5, 0.5])
    extended_std.extend([0.5, 0.5, 0.5])

    # 创建标准化transform并应用
    normalize = T.Normalize(mean=extended_mean, std=extended_std)
    batch_features = normalize(batch_features)

    return batch_features


def create_optimizer(model, optimizer_type='adamw'):
    '''
    根据配置创建优化器 AdamW or SGD
    '''
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE_SGD, momentum=config.SGD_MOMENTUM, weight_decay=config.SGD_WEIGHT_DECAY, nesterov=config.SGD_NESTEROV)
        print(f"使用SGD优化器 - 学习率: {config.LEARNING_RATE_SGD}, 动量: {config.SGD_MOMENTUM}, "
              f"Nesterov: {config.SGD_NESTEROV}, 权重衰减: {config.SGD_WEIGHT_DECAY}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE_ADAM, weight_decay=config.WEIGHT_DECAY, betas=config.BETAS, eps=config.EPS)
        print(f"使用AdamW优化器 - 学习率: {config.LEARNING_RATE_ADAM}, 权重衰减: {config.WEIGHT_DECAY}, "
              f"Betas: {config.BETAS}, Eps: {config.EPS}")

    return optimizer


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, eta_min):
    '''
    学习率调度策略
    '''
    # 检查是否使用SGD优化器且启用StepLR
    if (config.OPTIMIZER_TYPE.lower() == 'sgd' and config.SGD_USE_STEP_LR):
        step_size = config.SGD_STEP_SIZE
        gamma = config.SGD_GAMMA
        print(f"SGD使用StepLR - 步长: {step_size}, 衰减因子: {gamma}")
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        print(f"使用余弦退火学习率调度策略")

    # 余弦退火
    strategy = config.COSINE_STRATEGY
    if strategy == 'restart': # 余弦重启策略：每T轮重启一次
        restart_t = config.COSINE_RESTART_T
        restart_mult = config.COSINE_RESTART_MULT

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # 预热阶段：线性增长
                return config.WARMUP_FACTOR + (1.0 - config.WARMUP_FACTOR) * epoch / warmup_epochs
            else:
                # 余弦重启阶段
                cos_epoch = epoch - warmup_epochs
                restart_epoch = cos_epoch % restart_t
                restart_count = cos_epoch // restart_t
                current_lr = eta_min + (1 - eta_min) * 0.5 * (1 + np.cos(np.pi * restart_epoch / restart_t))
                return current_lr * (restart_mult**restart_count) # 每次重启后学习率乘以倍数

    elif strategy == 'warm_restart': # 热重启策略：重启时学习率逐渐降低
        restart_t = config.COSINE_RESTART_T

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # 预热阶段：线性增长
                return config.WARMUP_FACTOR + (1.0 - config.WARMUP_FACTOR) * epoch / warmup_epochs
            else:
                # 热重启阶段
                cos_epoch = epoch - warmup_epochs
                restart_epoch = cos_epoch % restart_t
                restart_count = cos_epoch // restart_t
                # 每次重启后最小学习率逐渐降低
                current_eta_min = eta_min * (0.9**restart_count)
                return current_eta_min + (1 - current_eta_min) * 0.5 * (1 + np.cos(np.pi * restart_epoch / restart_t))

    else: # 标准余弦退火策略
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # 预热阶段：线性增长
                return config.WARMUP_FACTOR + (1.0 - config.WARMUP_FACTOR) * epoch / warmup_epochs
            else:
                # 余弦退火阶段
                cos_epoch = epoch - warmup_epochs
                cos_total = total_epochs - warmup_epochs
                return eta_min + (1 - eta_min) * 0.5 * (1 + np.cos(np.pi * cos_epoch / cos_total))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
