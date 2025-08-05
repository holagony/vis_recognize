import torch

# --- 设备配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 深度分支预训练模型 ---
USE_SIMPLE_DEPTH = True  # True: 使用轻量级深度分支, False: 使用DPT分支
SIMPLE_DEPTH_MODEL_PATH = r'D:\Project\traffic\app\depth_scene.pth'

# --- 数据路径 ---
TRAIN_DATA_ROOT = r'D:\Project\traffic\交接\highway_train_data\highway_train_data'
VAL_DATA_ROOT = r'D:\Project\traffic\交接\highway_validate_data'

# --- 输出路径 ---
MODEL_OUTPUT_DIR = r'D:\Project\traffic\app\data'
INFERENCE_RESULT_DIR = r'D:\Project\traffic\app\data'

# --- 图像预处理 ---
TARGET_IMG_HEIGHT = 256
TARGET_IMG_WIDTH = 256
TARGET_INPUT_SIZE = (TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH) # (H, W)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# --- 训练超参数 ---
BATCH_SIZE = 12
GRADIENT_ACCUMULATION_STEPS = 3
EPOCHS = 60
LEARNING_RATE = 2e-4
NUM_CLASSES = 5

# --- 优化器超参数 ---
WEIGHT_DECAY = 0.01               # 适中的权重衰减，平衡正则化和性能
BETAS = (0.9, 0.999)              # Adam动量参数
EPS = 1e-8                        # Adam epsilon

# --- 学习率调度超参数 ---
WARMUP_EPOCHS = 3                 # 减少预热轮数
WARMUP_FACTOR = 0.2               # 提高起始因子，避免过度保守
LR_SCHEDULE_TYPE = 'cosine_warmup' # 使用带预热的余弦调度
ETA_MIN = 2e-5                    # 最小学习率，避免过小影响收敛

# --- 正则化超参数 ---
GRADIENT_CLIP_NORM = 1.0          # 标准梯度裁剪阈值，适合稳定训练
LABEL_SMOOTHING = 0.1             # 标签平滑

# --- 特征分支参数 ---
TRANSMISSION_OMEGA = 0.95
TRANSMISSION_PATCH_SIZE = 5
TRANSMISSION_GUIDED_RADIUS = 60
TRANSMISSION_GUIDED_EPS = 1e-3
DETAIL_GUIDED_RADIUS = 8
DETAIL_GUIDED_EPS = 0.02**2
SPECTRAL_ENHANCEMENT_FACTOR = 2.2      # 轻微增加光谱增强

# --- 损失函数超参数 ---
FOCAL_GAMMA = 2.0                      # 标准Focal Loss的gamma，适合中等不平衡
FOCAL_ALPHA = 1.0                     # 标准Focal Loss alpha参数
WEIGHT_MODE = 'sqrt_balanced'          # 平方根平衡权重，缓解不平衡影响
SMOOTH_FACTOR = 0.1                    # 减少平滑因子，保持类别区分度

# --- channel数量 --- 
SFRB_OUT_CHANNELS = 64    # 11->64，初始特征提取
NUM_MSFB_BLOCKS = 2       # 2个MSFB块
MSFB_CHANNEL_MULTIPLIERS = [2, 4]  # MSFB1: 64*2=128, MSFB2: 64*4=256
GFFB_OUT_CHANNELS = 512   # 最终输出512通道


def get_model_kwargs():
    """获取模型的关键字参数"""
    return {
        'num_visibility_levels': NUM_CLASSES,
        'sfrb_out_channels': SFRB_OUT_CHANNELS,
        'num_msfb_blocks': NUM_MSFB_BLOCKS,
        'gffb_out_channels': GFFB_OUT_CHANNELS,
        'img_size_tuple': TARGET_INPUT_SIZE,
        'device': DEVICE,
        'use_simple_depth': USE_SIMPLE_DEPTH,
    }

