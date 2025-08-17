import os
import random
import logging
import psutil
import numpy as np
import torch
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
