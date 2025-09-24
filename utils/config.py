import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cuda:1")

# ==================== 数据配置 ====================
# 数据路径
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(CURRENT_DIR, 'results')
DATA_DIR = os.path.join(CURRENT_DIR, './img_data/data')
TRAIN_DATA_ROOT = os.path.join(DATA_DIR, 'train')
VAL_DATA_ROOT = os.path.join(DATA_DIR, 'val')
TEST_DATA_ROOT = os.path.join(DATA_DIR, 'test')
MODEL_OUTPUT_DIR = os.path.join(RESULT_DIR, 'models')

# 图像预处理
TARGET_INPUT_SIZE = (256, 256)
TRAIN_RESIZE_MODE = 'random_crop'  # 'direct', 'pad', 'center_crop', 'random_crop'
VAL_RESIZE_MODE = 'center_crop'
USE_AUGMENTATION = False  # 数据增强

# ==================== 模型配置 ====================
# 支持的模型类型:
# ResNet系列: 'resnet18', 'resnet34', 'resnet50'
# ResNet+SupCon: 'resnet18_supcon', 'resnet34_supcon', 'resnet50_supcon'
# EfficientNet系列: 'efficientnet_b0', 'efficientnet_b1'
# EfficientNet+SupCon: 'efficientnet_b0_supcon', 'efficientnet_b1_supcon', 'efficientnet_b2_supcon'
# 其他: 'wuhan'
MODEL_TYPE = 'resnet34_supcon'
USE_SIMPLE_DEPTH = True  # True: MobileNet, False: DPT
SIMPLE_DEPTH_MODEL_PATH = './model_hub/depth_scene.pth'

# ==================== 训练基础配置 ====================
# 基本训练参数
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 1  # 梯度累积
EPOCHS = 80
NUM_CLASSES = 5
GRADIENT_CLIP_NORM = 1.0  # 梯度裁剪

# 早停配置
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001  # 最小改进阈值

# WeightedRandomSampler相关
BALANCE_FACTOR = 0.3  # 控制采样平衡的比例
USE_SAMPLER_REPLACEMENT = False  # 是否采样有放回

# ==================== 优化器配置 ====================
# SGD 参数
LEARNING_RATE_SGD = 5e-2
SGD_MOMENTUM = 0.9  # SGD动量参数
SGD_WEIGHT_DECAY = 1e-4  # SGD权重衰减（通常比AdamW小）
SGD_NESTEROV = False  # 是否使用Nesterov动量

# AdamW 参数
LEARNING_RATE_ADAM = 1e-4
WEIGHT_DECAY = 1e-3  # 默认1e-2
BETAS = (0.9, 0.999)
EPS = 1e-8

# ==================== 学习率调度配置 ====================
# 预热参数
WARMUP_EPOCHS = 5  # 预热轮数
WARMUP_FACTOR = 0.2  # 预热起始因子：从0.2倍学习率开始
ETA_MIN = 1e-5  # 最小学习率

# 余弦退火策略
COSINE_STRATEGY = 'standard'  # 'standard': 标准余弦退火, 'restart': 余弦重启, 'warm_restart': 热重启
COSINE_RESTART_T = 10  # 重启周期（仅当使用restart策略时）
COSINE_RESTART_MULT = 2.0  # 重启后学习率倍数

# SGD学习率调度
SGD_USE_STEP_LR = True  # StepLR调度器
SGD_STEP_SIZE = 5  # StepLR的步长
SGD_GAMMA = 0.7  # StepLR的学习率衰减因子

# ==================== 损失函数配置 ====================
# Focal Loss 参数
FOCAL_GAMMA = 2
FOCAL_ALPHA = [0.3, 0.4, 0.8, 1.0, 0.9]

# 根据类别计算权重
WEIGHT_MODE = 'balanced'
SMOOTH_FACTOR = 0.05  # 权重平滑
LABEL_SMOOTHING = 0.05  # 标签平滑

# SupCon loss
SUPCON_TEMPERATURE = 0.07  # 温度参数
SUPCON_WEIGHT = 1  # 0.6

# CE loss
CE_WEIGHT = 0.4  # 0.4 / 0.5

# Dice Loss
DICE_SMOOTH = 1e-6  # Dice Loss平滑因子
DICE_WEIGHT = 0.5  # Dice Loss权重

# ==================== 特征提取配置 ====================
# 传输通道参数
# 512x512 path_size=9. radius=20
TRANSMISSION_OMEGA = 0.95
TRANSMISSION_PATCH_SIZE = 7
TRANSMISSION_GUIDED_RADIUS = 15
TRANSMISSION_GUIDED_EPS = 1e-3

# ==================== 配置说明 ====================
# 学习率调度说明：
# 1. 预热阶段(0-5轮)：学习率从 2e-5 线性增长到 1e-4
# 2. 余弦退火阶段(5-60轮)：学习率按余弦函数从 1e-4 衰减到 5e-6
# 3. 总学习率变化范围：2e-5 → 1e-4 → 5e-6
# 4. 可选：使用余弦重启策略，每10轮重启一次学习率

# Focal Loss alpha权重说明：
# - FOCAL_ALPHA = None: 自动计算类别特定权重（推荐）
# - FOCAL_ALPHA = [0.1, 0.3, 0.8, 1.0, 0.5]: 手动设置每个类别的权重
# - 权重范围：[0.1, 1.0]，数值越大表示对该类别越关注
# - 建议：少数类别（如类别2、3）设置更高的权重（0.8-1.0）

# SGD优化器配置说明：
# 1. 基本参数：
#    - OPTIMIZER_TYPE: 选择优化器类型 ('adamw' 或 'sgd')
#    - SGD_MOMENTUM: 动量参数，通常设置为0.9
#    - SGD_NESTEROV: 是否使用Nesterov动量，通常为True
#    - SGD_WEIGHT_DECAY: 权重衰减，通常比AdamW小(1e-4)
#
# 2. 学习率调度：
#    - SGD_USE_STEP_LR: 是否使用StepLR调度器（推荐用于SGD）
#    - SGD_STEP_SIZE: StepLR步长，每多少轮衰减一次学习率
#    - SGD_GAMMA: 学习率衰减因子，每次衰减为原来的多少倍
#
# 3. 使用建议：
#    - SGD通常需要更大的学习率（如1e-2到1e-1）
#    - 配合StepLR或MultiStepLR调度器效果更好
#    - 对于计算机视觉任务，SGD+动量通常能获得更好的泛化性能
#    - 训练初期可能需要更多轮次才能收敛
