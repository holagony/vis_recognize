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
WEIGHT_DECAY = 0.005              # 减少权重衰减，避免过度正则化
BETAS = (0.9, 0.999)              # Adam动量参数
EPS = 1e-8                        # Adam epsilon

# --- 学习率调度超参数 ---
WARMUP_EPOCHS = 5                 # 学习率预热轮数
WARMUP_FACTOR = 0.1               # 预热起始因子
LR_SCHEDULE_TYPE = 'cosine_warmup' # 使用带预热的余弦调度
T_MAX = 55                        # 余弦调度的最大轮数（总轮数-预热轮数）
ETA_MIN = 1e-6                    # 最小学习率

# --- 正则化超参数 ---
GRADIENT_CLIP_NORM = 2.0          # 增加梯度裁剪阈值
LABEL_SMOOTHING = 0.1             # 标签平滑

# --- 早停和模型保存 ---
EARLY_STOP_PATIENCE = 8           # 早停耐心值
SAVE_CHECKPOINT_EVERY = 3         # 每3轮保存检查点
MIN_DELTA = 0.001                 # 早停最小改进阈值

# --- 特征分支参数 ---
TRANSMISSION_OMEGA = 0.95
TRANSMISSION_PATCH_SIZE = 5
TRANSMISSION_GUIDED_RADIUS = 60
TRANSMISSION_GUIDED_EPS = 1e-3
DETAIL_GUIDED_RADIUS = 8
DETAIL_GUIDED_EPS = 0.02**2
SPECTRAL_ENHANCEMENT_FACTOR = 2.2      # 轻微增加光谱增强

# --- 损失函数超参数 ---
FOCAL_GAMMA = 2.5                      # 增加Focal Loss的gamma
FOCAL_ALPHA = 1.0                      # 标准alpha，不使用auto权重
WEIGHT_MODE = 'sqrt_balanced'          # 平方根平衡权重（备用）
SMOOTH_FACTOR = 0.25                   # 权重平滑因子（备用）

# --- 数据增强超参数 ---
AUGMENTATION_PROB = 0.6               # 数据增强概率
BRIGHTNESS_RANGE = 0.15               # 亮度调整范围
CONTRAST_RANGE = 0.15                 # 对比度调整范围
SATURATION_RANGE = 0.1                # 饱和度调整范围
HUE_RANGE = 0.05                      # 色调调整范围

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

