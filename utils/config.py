import torch

# --- 设备配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 5
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数，实际批次大小 = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

# --- 特征分支参数 ---
TRANSMISSION_OMEGA = 0.95
TRANSMISSION_PATCH_SIZE = 5
TRANSMISSION_GUIDED_RADIUS = 60
TRANSMISSION_GUIDED_EPS = 1e-3
DETAIL_GUIDED_RADIUS = 8
DETAIL_GUIDED_EPS = 0.02**2
SPECTRAL_ENHANCEMENT_FACTOR = 2.0

# --- 深度分支配置 ---
USE_SIMPLE_DEPTH = True  # True: 使用轻量级深度分支, False: 使用DPT分支
SIMPLE_DEPTH_MODEL_PATH = r'D:\Project\traffic\app\depth_scene.pth'

# --- channel数量 ---
SFRB_OUT_CHANNELS = 128
NUM_MSFB_BLOCKS = 2
GFFB_OUT_CHANNELS = 512


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

