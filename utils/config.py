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

# 数据增强配置
AUGMENTATION_CONFIG = {
    'enable_advanced_aug': True,      # 是否启用高级增强
    'geometric_aug_prob': 0.3,        # 几何变换概率
    'weather_sim_prob': 0.2,          # 天气模拟概率
    'noise_aug_prob': 0.15,           # 噪声增强概率
    
    # 第3类特殊增强配置
    'class_3_enhancement': {
        'brightness_prob': 0.7,        # 亮度调整概率
        'contrast_prob': 0.6,          # 对比度调整概率
        'sharpening_prob': 0.4,        # 锐化概率
        'blur_prob': 0.3,              # 模糊概率
        'color_prob': 0.4,             # 色彩增强概率
        'rotation_prob': 0.4,          # 旋转概率
        'crop_prob': 0.3,              # 裁剪概率
    }
}

# 模型配置
USE_SIMPLE_DEPTH = True  # True: 使用轻量级深度分支, False: 使用DPT分支
SIMPLE_DEPTH_MODEL_PATH = './model_hub/depth_scene.pth'
MODEL_TYPE = 'resnet'  # 'vismfn' 或 'resnet'

# 训练超参数
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 4 # 梯度累积
EPOCHS = 80
NUM_CLASSES = 5
GRADIENT_CLIP_NORM = 1.0 # 梯度裁剪 5.0

# AdamW 参数
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
EPS = 1e-8

# warmup + 余弦退火 参数
WARMUP_EPOCHS = 4
WARMUP_FACTOR = 0.1
ETA_MIN = 1e-6

# 早停配置
EARLY_STOPPING_PATIENCE = 10  # 早停耐心值
EARLY_STOPPING_MIN_DELTA = 0.005  # 最小改进阈值

# 损失函数超参数
FOCAL_GAMMA = 2.0 # 标准Focal Loss的gamma，适合中等不平衡
FOCAL_ALPHA = 1.0 # 标准Focal Loss alpha参数
WEIGHT_MODE = 'sqrt_balanced' # 平方根平衡权重，缓解不平衡影响
SMOOTH_FACTOR = 0.05 # 减少平滑因子，保持类别区分度
LABEL_SMOOTHING = 0.03 # 不平衡任务需减少平滑，避免主导类别的概率泄露到罕见类别。


# vismfn channel数量
SFRB_OUT_CHANNELS = 64    # 11->64，初始特征提取
NUM_MSFB_BLOCKS = 2       # 2个MSFB块
MSFB_CHANNEL_MULTIPLIERS = [2, 4]  # MSFB1: 64*2=128, MSFB2: 64*4=256
GFFB_OUT_CHANNELS = 512   # 最终输出512通道

# 特征提取块参数
TRANSMISSION_OMEGA = 0.95
TRANSMISSION_PATCH_SIZE = 5
TRANSMISSION_GUIDED_RADIUS = 25
TRANSMISSION_GUIDED_EPS = 1e-3
DETAIL_GUIDED_RADIUS = 8
DETAIL_GUIDED_EPS = 0.02**2
SPECTRAL_ENHANCEMENT_FACTOR = 2.2 # 轻微增加光谱增强


def get_model_kwargs():
    '''
    获取模型的关键字参数
    '''
    if MODEL_TYPE == 'vismfn':
        return {
            'num_visibility_levels': NUM_CLASSES,
            'sfrb_out_channels': SFRB_OUT_CHANNELS,
            'num_msfb_blocks': NUM_MSFB_BLOCKS,
            'gffb_out_channels': GFFB_OUT_CHANNELS,
            'img_size_tuple': TARGET_INPUT_SIZE,
            'device': DEVICE,
            'use_simple_depth': USE_SIMPLE_DEPTH,
        }
    elif MODEL_TYPE == 'resnet':
        return {
            'num_classes': NUM_CLASSES
        }
    else:
        raise ValueError(f"不支持的模型类型: {MODEL_TYPE}")

