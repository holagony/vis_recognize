# EfficientNet模型说明

本目录包含了针对能见度识别任务优化的EfficientNet模型实现。

## 支持的模型

### 基础模型
- **EfficientNet-B0**: 轻量级模型，参数量约5.3M
- **EfficientNet-B1**: 平衡性能与效率，参数量约7.8M
- **EfficientNet-B2**: 更高精度，适合512×512输入，参数量约9.2M

### SupCon对比学习模型
- **EfficientNet-B0+SupCon**: B0 + 监督对比学习
- **EfficientNet-B1+SupCon**: B1 + 监督对比学习
- **EfficientNet-B2+SupCon**: B2 + 监督对比学习

## 配置使用

在 `utils/config.py` 中设置 `MODEL_TYPE`：

```python
# 基础EfficientNet模型
MODEL_TYPE = 'efficientnet_b0'     # EfficientNet-B0
MODEL_TYPE = 'efficientnet_b1'     # EfficientNet-B1

# EfficientNet + SupCon模型
MODEL_TYPE = 'efficientnet_b0_supcon'  # B0 + SupCon
MODEL_TYPE = 'efficientnet_b1_supcon'  # B1 + SupCon
MODEL_TYPE = 'efficientnet_b2_supcon'  # B2 + SupCon
```

## 推荐配置

### EfficientNet-B0
```python
MODEL_TYPE = 'efficientnet_b0_supcon'
BATCH_SIZE = 128
LEARNING_RATE_ADAM = 1e-3  # 从零训练需要更高学习率
TARGET_INPUT_SIZE = (384, 384)
EPOCHS = 200  # 从零训练需要更多轮次
```

### EfficientNet-B1 (推荐)
```python
MODEL_TYPE = 'efficientnet_b1_supcon'
BATCH_SIZE = 128
LEARNING_RATE_ADAM = 8e-4  # 从零训练需要更高学习率
TARGET_INPUT_SIZE = (384, 384)
EPOCHS = 200  # 从零训练需要更多轮次
```

### EfficientNet-B2 (高分辨率)
```python
MODEL_TYPE = 'efficientnet_b2_supcon'
BATCH_SIZE = 48
LEARNING_RATE_ADAM = 5e-4  # 从零训练需要更高学习率
TARGET_INPUT_SIZE = (512, 512)
EPOCHS = 250  # 从零训练需要更多轮次
```

## 特性

### 多通道输入支持
- 自动适配11通道输入（RGB + 8个特征通道）
- 智能权重初始化，从零开始训练

### SupCon对比学习
- 投影维度：128（可配置）
- 温度参数：0.07
- 联合训练：对比学习 + 交叉熵损失

### 内存优化
- 支持混合精度训练（AMP）
- 参数量相比ResNet显著减少
- 更高的计算效率

## 性能对比

| 模型 | 参数量 | 显存占用 | 推荐Batch Size | 适用场景 |
|------|--------|----------|----------------|----------|
| EfficientNet-B0 | 5.3M | 低 | 128-160 | 快速训练/推理 |
| EfficientNet-B1 | 7.8M | 中等 | 128 | 平衡性能 |
| EfficientNet-B2 | 9.2M | 中等 | 48-64 | 高分辨率 |
| ResNet34 | 21.8M | 高 | 96 | 传统方案 |
| ResNet50 | 25.6M | 很高 | 64 | 高精度需求 |

## 依赖要求

```bash
pip install timm>=0.9.0
pip install torch>=1.12.0
pip install torchvision>=0.13.0
```

## 测试

运行测试脚本验证模型：

```bash
python test_efficientnet.py
```

## 训练建议

### 数据增强
```python
USE_AUGMENTATION = True  # 强烈推荐
```

### 优化器设置
```python
# AdamW优化器（推荐）
LEARNING_RATE_ADAM = 1e-3  # 从零训练使用更高学习率
WEIGHT_DECAY = 1e-3

# 学习率调度
WARMUP_EPOCHS = 10  # 从零训练增加warmup轮次
ETA_MIN = 1e-6
```

### SupCon参数
```python
SUPCON_TEMPERATURE = 0.05  # 可以尝试更低的温度
SUPCON_WEIGHT = 1.0
CE_WEIGHT = 0.5
```

## 注意事项

1. **从零训练**: 模型不使用预训练权重，完全从随机初始化开始训练
2. **显存管理**: 建议开启混合精度训练节省显存
3. **批次大小**: 根据显存大小调整，EfficientNet相比ResNet可以使用更大的批次
4. **收敛速度**: 由于从零训练，需要更多训练轮次和更高学习率
5. **训练轮次**: 建议200-300个epoch，比使用预训练权重需要更长时间

## 故障排除

### 常见问题

1. **ImportError: No module named 'timm'**
   ```bash
   pip install timm
   ```

2. **CUDA out of memory**
   - 减少 `BATCH_SIZE`
   - 开启混合精度训练
   - 使用梯度累积

3. **模型加载失败**
   - 检查网络连接（下载预训练权重）
   - 确认模型类型配置正确

### 性能优化

1. **开启混合精度**
   ```python
   # 在训练脚本中
   scaler = GradScaler()
   ```

2. **使用更大批次**
   ```python
   BATCH_SIZE = 128  # EfficientNet可以支持更大批次
   ```

3. **调整学习率**
   ```python
   # 批次大时可以适当提高学习率
   LEARNING_RATE_ADAM = 1.5e-4
   ```