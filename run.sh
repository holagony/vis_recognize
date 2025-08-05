#!/bin/bash
# 类别不平衡处理策略训练命令集

# ===========================================
# 方案1: 纯Focal Loss（推荐基线）
# 特点: 让Focal Loss自动处理不平衡，避免双重矫正
# 适用: 轻中度不平衡，想要最佳平衡性能
# ===========================================
nohup python train.py --loss_type focal &

# ===========================================
# 方案2: Focal Loss + Weighted Sampler（双重保障）
# 特点: 采样平衡 + 困难样本关注，机制互补
# 适用: 极度不平衡（比例>50:1），多数类占绝对优势
# ===========================================
# nohup python train.py --loss_type focal --weighted_sampler &

# ===========================================
# 方案3: CrossEntropy + Weighted Loss（传统权重方法）
# 特点: 直接调整类别权重，简单直接
# 适用: 中度不平衡，需要精确控制各类别权重
# ===========================================
# nohup python train.py --loss_type crossentropy --weighted_loss &


# ===========================================
# 方案4: CrossEntropy + Weighted Sampler（经典采样平衡）
# 特点: 通过采样平衡数据分布
# 适用: 传统方法，数据量充足时的选择
# ===========================================
# nohup python train.py --loss_type crossentropy --weighted_sampler &