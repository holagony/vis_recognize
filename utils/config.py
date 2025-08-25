import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cuda:1")

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
RESULT_DIR = os.path.join(CURRENT_DIR, 'results')

# 数据路径
TRAIN_DATA_ROOT = os.path.join(DATA_DIR, 'train')
VAL_DATA_ROOT = os.path.join(DATA_DIR, 'val')
TEST_DATA_ROOT = os.path.join(DATA_DIR, 'test')

# 输出路径
MODEL_OUTPUT_DIR = os.path.join(RESULT_DIR, 'models')
INFERENCE_RESULT_DIR = os.path.join(RESULT_DIR, 'inference')

# 图像预处理
TARGET_INPUT_SIZE = (256, 256) # (H, W)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
DIRECT_RESIZE = True  # True: 直接resize到目标尺寸, False: 保持长宽比+填充
USE_AUGMENTATION = False  # 是否启用数据增强

# 模型配置
USE_SIMPLE_DEPTH = True  # True: 使用轻量级深度分支, False: 使用DPT分支
SIMPLE_DEPTH_MODEL_PATH = './model_hub/depth_scene.pth'
MODEL_TYPE = 'supcon'  # resnet / supcon / wuhan

# 训练超参数
BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1 # 梯度累积
EPOCHS = 60
NUM_CLASSES = 5
GRADIENT_CLIP_NORM = 1.0 # 梯度裁剪 5.0

# WeightedRandomSampler采样
BALANCE_FACTOR = 0.3  # 平衡因子，控制采样策略的激进程度
USE_SAMPLER_REPLACEMENT = False  # 是否使用替换采样

# AdamW 参数
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3  # 默认1e-2
BETAS = (0.9, 0.999)
EPS = 1e-8

# warmup + 余弦退火 参数
WARMUP_EPOCHS = 5        # 预热轮数
WARMUP_FACTOR = 0.2      # 预热起始因子：从0.1倍学习率开始
ETA_MIN = 1e-5           # 最小学习率

# 余弦退火策略选择
COSINE_STRATEGY = 'standard'  # 'standard': 标准余弦退火, 'restart': 余弦重启, 'warm_restart': 热重启
COSINE_RESTART_T = 10         # 重启周期（仅当使用restart策略时）
COSINE_RESTART_MULT = 2.0     # 重启后学习率倍数

# 学习率调度说明：
# 1. 预热阶段(0-3轮)：学习率从 5e-6 线性增长到 1e-4
# 2. 余弦退火阶段(3-60轮)：学习率按余弦函数从 1e-4 衰减到 5e-6
# 3. 总学习率变化范围：5e-6 → 1e-4 → 5e-6
# 4. 可选：使用余弦重启策略，每10轮重启一次学习率

# 早停配置
EARLY_STOPPING_PATIENCE = 10  # 早停耐心值，从5增加到15
EARLY_STOPPING_MIN_DELTA = 0.001  # 最小改进阈值，从0.005降低到0.001

# 损失函数超参数
FOCAL_GAMMA = 2 # 增加gamma值，从2.0到3.0，更关注难分类样本
FOCAL_ALPHA = [0.3, 0.4, 0.8, 1.0, 0.9]
WEIGHT_MODE = 'balanced'
SMOOTH_FACTOR = 0.05  # 降低平滑因子，从0.03到0.01，保持权重差异
LABEL_SMOOTHING = 0.05  # 增加标签平滑，从0.01到0.05，提高泛化能力

# Focal Loss alpha权重说明：
# - FOCAL_ALPHA = None: 自动计算类别特定权重（推荐）
# - FOCAL_ALPHA = [0.1, 0.3, 0.8, 1.0, 0.5]: 手动设置每个类别的权重
# - 权重范围：[0.1, 1.0]，数值越大表示对该类别越关注
# - 建议：少数类别（如类别2、3）设置更高的权重（0.8-1.0）

# 特征提取块参数
TRANSMISSION_OMEGA = 0.95
TRANSMISSION_PATCH_SIZE = 5
TRANSMISSION_GUIDED_RADIUS = 25
TRANSMISSION_GUIDED_EPS = 1e-3
DETAIL_GUIDED_RADIUS = 8
DETAIL_GUIDED_EPS = 0.0004
SPECTRAL_ENHANCEMENT_FACTOR = 2.2 # 轻微增加光谱增强

# SupCon 训练配置
SUPCON_TEMPERATURE = 0.07  # 温度参数
SUPCON_WEIGHT = 0.5        # SupCon 损失权重
CE_WEIGHT = 0.5            # 交叉熵损失权重