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
USE_AUGMENTATION = True  # 是否启用数据增强

# 模型配置
USE_SIMPLE_DEPTH = True  # True: 使用轻量级深度分支, False: 使用DPT分支
SIMPLE_DEPTH_MODEL_PATH = './model_hub/depth_scene.pth'
MODEL_TYPE = 'resnet'  # 'vismfn' 或 'resnet'

# ResNet空洞卷积配置
RESNET_USE_DILATION = True  # 是否启用空洞卷积
RESNET_DILATION_RATES = [1, 1, 2, 4]  # [layer1, layer2, layer3, layer4] 的空洞率
# 空洞率说明：
# - layer1, layer2: 保持dilation=1，维持高分辨率特征
# - layer3: dilation=2，增加感受野
# - layer4: dilation=4，进一步增加感受野

# 训练超参数
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 4 # 梯度累积
EPOCHS = 60
NUM_CLASSES = 5
GRADIENT_CLIP_NORM = 1.0 # 梯度裁剪 5.0

# AdamW 参数
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
EPS = 1e-8

# warmup + 余弦退火 参数
WARMUP_EPOCHS = 4        # 预热轮数：建议为总轮数的5-10%
WARMUP_FACTOR = 0.1      # 预热起始因子：从0.1倍学习率开始
ETA_MIN = 1e-5           # 最小学习率：建议为基础学习率的1/10到1/100

# 学习率调度说明：
# 1. 预热阶段(0-4轮)：学习率从 1e-5 线性增长到 1e-4
# 2. 余弦退火阶段(4-60轮)：学习率按余弦函数从 1e-4 衰减到 1e-5
# 3. 总学习率变化范围：1e-5 → 1e-4 → 1e-5

# 早停配置
EARLY_STOPPING_PATIENCE = 5  # 早停耐心值
EARLY_STOPPING_MIN_DELTA = 0.005  # 最小改进阈值

# 损失函数超参数
FOCAL_GAMMA = 2.0 # 标准Focal Loss的gamma，适合中等不平衡
FOCAL_ALPHA = 1.0 # 标准Focal Loss alpha参数
WEIGHT_MODE = 'sqrt_balanced' # 平方根平衡权重，缓解不平衡影响
SMOOTH_FACTOR = 0.05 # 减少平滑因子，保持类别区分度
LABEL_SMOOTHING = 0.01 # 不平衡任务需减少平滑，避免主导类别的概率泄露到罕见类别。

# 特征提取块参数
TRANSMISSION_OMEGA = 0.95
TRANSMISSION_PATCH_SIZE = 5
TRANSMISSION_GUIDED_RADIUS = 25
TRANSMISSION_GUIDED_EPS = 1e-3
DETAIL_GUIDED_RADIUS = 8
DETAIL_GUIDED_EPS = 0.0004
SPECTRAL_ENHANCEMENT_FACTOR = 2.2 # 轻微增加光谱增强


# vismfn channel数量
SFRB_OUT_CHANNELS = 64    # 11->64，初始特征提取
NUM_MSFB_BLOCKS = 2       # 2个MSFB块
MSFB_CHANNEL_MULTIPLIERS = [2, 4]  # MSFB1: 64*2=128, MSFB2: 64*4=256
GFFB_OUT_CHANNELS = 512   # 最终输出512通道