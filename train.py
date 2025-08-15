import os
import glob
import logging
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast

try:
   from torch import GradScaler        # torch >= 2.3
except ImportError:
   from torch.cuda.amp import GradScaler

from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from models.vismfn.model import VisMFN
from utils.loss import create_loss_function
from utils import config
from utils.utils import set_seed, setup_logging, get_memory_usage
from utils.metric import calculate_metrics
from utils.loss import create_loss_function
from datasets.vis_dataloader import get_dataloader
from models.resnet.resnet_cbam import ResNet, resnet18_cbam 

def train_one_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=1, epoch=None, scaler=None):
    """
    训练一个epoch，支持梯度累积和混合精度
    """
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    optimizer.zero_grad()  # 在epoch开始时清零梯度

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}' if epoch is not None else 'Training')
    for batch_idx, (inputs, labels) in enumerate(pbar):
        if inputs is None or labels is None:
            continue
            
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 混合精度前向传播
        if scaler is not None:
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # 缩放损失以匹配累积步数
                loss = loss / accumulation_steps
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
        else:
            # 标准精度前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 缩放损失以匹配累积步数
            loss = loss / accumulation_steps
            loss.backward()
        
        # 累积梯度
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_NORM)
                optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps  # 恢复原始损失值用于记录
        _, predicted = torch.max(outputs, 1)
        
        # 收集预测和标签用于详细指标计算
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条 - 使用更适合不平衡任务的指标
        current_loss = running_loss / (batch_idx + 1)
        
        # 计算当前批次的平衡准确率
        if len(all_predictions) > 0:
            # 确保预测标签在合法范围内，避免sklearn警告
            clipped_predictions = np.clip(np.array(all_predictions), 0, config.NUM_CLASSES - 1)
            current_balanced_acc = 100 * balanced_accuracy_score(all_labels, clipped_predictions)
            current_overall_acc = 100 * np.mean(np.array(all_labels) == clipped_predictions)
        else:
            current_balanced_acc = 0.0
            current_overall_acc = 0.0
            
        pbar.set_postfix({'Loss': f'{current_loss:.4f}',
                          'Bal_Acc': f'{current_balanced_acc:.2f}%',
                          'Overall_Acc': f'{current_overall_acc:.2f}%',
                          'Accum': f'{((batch_idx + 1) % accumulation_steps) + 1}/{accumulation_steps}'})
    
    # 处理最后一个不完整的累积批次
    if len(dataloader) % accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()
    
    # 计算详细指标
    metrics = calculate_metrics(all_labels, all_predictions, config.NUM_CLASSES)
    avg_loss = running_loss / len(dataloader)
    
    return avg_loss, metrics


def validate(model, dataloader, criterion, device):
    """
    验证模型
    """
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            if inputs is None or labels is None:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            # 收集预测和标签
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算详细指标
    metrics = calculate_metrics(all_labels, all_predictions, config.NUM_CLASSES)
    avg_loss = running_loss / len(dataloader)
    
    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, accuracy, best_accuracy, model_config, save_path):
    """
    保存检查点
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'best_accuracy': best_accuracy,
        'model_config': model_config
    }, save_path)


def main():
    parser = argparse.ArgumentParser(description='Vis Training')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--loss_type', type=str, choices=['crossentropy', 'focal'], default='crossentropy')
    parser.add_argument('--weighted_sampler', action='store_true', help='是否使用加权采样器')
    parser.add_argument('--weighted_loss', action='store_true', help='是否在损失函数中使用类别权重')

    # 有default
    parser.add_argument('--weight_mode', type=str, choices=['balanced', 'sqrt_balanced', 'log_balanced'], default=config.WEIGHT_MODE)
    parser.add_argument('--seed', type=int, default=3407, help='随机种子')
    args = parser.parse_args()
    
    # 设置全局随机种子
    set_seed(args.seed)
    
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    logger = setup_logging(config.MODEL_OUTPUT_DIR) # 设置日志
    tb_writer = SummaryWriter(log_dir=os.path.join(config.MODEL_OUTPUT_DIR, 'tensorboard'))

    # 加载数据和创建数据集
    train_loader, val_loader = get_dataloader(config.TRAIN_DATA_ROOT, config.VAL_DATA_ROOT, config.USE_AUGMENTATION, weighted_sampler=True)

    logger.info(f"随机种子: {args.seed}")
    logger.info(f"批次大小: {config.BATCH_SIZE}, 有效批次大小: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"加权策略: 采样器={args.weighted_sampler}, 损失权重={args.weighted_loss}")
    if args.weighted_loss:
        logger.info(f"权重模式: {args.weight_mode}, 平滑因子: {args.smooth_factor}")
    
    # 创建模型
    logger.info("正在初始化模型...")
    model_kwargs = config.get_model_kwargs()    
    model = VisMFN(**model_kwargs)
    model_config = model_kwargs.copy()
        
    # 避免双重加权的建议：如果使用加权采样器，建议不使用损失函数权重
    if args.weighted_sampler and args.weighted_loss:
        logger.warning("警告：同时使用加权采样器和损失函数权重可能导致双重加权，建议只使用其中一种")
    
    # 处理focal_alpha参数
    try:
        focal_alpha = float(args.focal_alpha)
    except ValueError:
        focal_alpha = args.focal_alpha  # 'auto'等字符串值
    
    # loss和优化器
    criterion = create_loss_function(
        train_labels, 
        loss_type=args.loss_type, 
        alpha=focal_alpha, 
        gamma=args.focal_gamma,
        use_weights=args.weighted_loss,
        weight_mode=args.weight_mode,
        smooth_factor=args.smooth_factor,
        label_smoothing=args.label_smoothing
        )
    # 使用配置文件中的优化器参数
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY, 
        betas=config.BETAS,
        eps=config.EPS)
    
    # 带预热的学习率调度器
    def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, eta_min):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # 预热阶段：线性增长
                return config.WARMUP_FACTOR + (1.0 - config.WARMUP_FACTOR) * epoch / warmup_epochs
            else:
                # 余弦退火阶段
                cos_epoch = epoch - warmup_epochs
                cos_total = total_epochs - warmup_epochs
                return eta_min + (1 - eta_min) * 0.5 * (1 + np.cos(np.pi * cos_epoch / cos_total))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scheduler = get_lr_scheduler(optimizer, config.WARMUP_EPOCHS, config.EPOCHS, config.ETA_MIN / config.LEARNING_RATE)
    
    # 混合精度训练
    try:
        scaler = GradScaler(device='cuda') if torch.cuda.is_available() else None
    except:
        scaler = GradScaler() if torch.cuda.is_available() else None
    
    # 训练变量
    start_epoch = 0
    best_accuracy = 0.0
    
    # 恢复训练
    if args.resume and os.path.exists(args.resume):
        logger.info(f"从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        logger.info(f"恢复训练从第 {start_epoch} 轮开始，当前最佳准确率: {best_accuracy:.4f}")
    
    # 将模型移动到设备
    model = model.to(config.DEVICE)
    logger.info(f"模型已移动到设备: {config.DEVICE}")
    
    # 在TensorBoard中可视化网络结构
    try:
        logger.info("正在生成网络结构图...")
        # 创建示例输入张量
        dummy_input = torch.randn(1, 3, config.TARGET_IMG_HEIGHT, config.TARGET_IMG_WIDTH).to(config.DEVICE)
        
        # 使用add_graph添加网络结构到TensorBoard
        tb_writer.add_graph(model, dummy_input)
        logger.info("网络结构图已添加到TensorBoard")
        logger.info(f"📊 启动TensorBoard查看网络结构：tensorboard --logdir={os.path.join(config.MODEL_OUTPUT_DIR, 'tensorboard')}")
        
        # 立即刷新TensorBoard
        tb_writer.flush()
        
    except Exception as e:
        logger.warning(f"生成网络结构图时出现错误: {e}")
        logger.warning("训练将继续进行，但网络结构图未生成")
    
    # 训练循环
    logger.info("开始训练...")
    for epoch in range(start_epoch, config.EPOCHS):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE, config.GRADIENT_ACCUMULATION_STEPS, epoch, scaler)
        val_loss, val_metrics = validate(model, val_loader, criterion, config.DEVICE)
        scheduler.step() # update learning rate
        
        # 结果记录 - 使用平衡准确率作为主要指标
        tb_writer.add_scalar('Loss/Train', train_loss, epoch)
        tb_writer.add_scalar('Loss/Validation', val_loss, epoch)
        tb_writer.add_scalar('Accuracy/Train_Overall', train_metrics['overall_accuracy'], epoch)
        tb_writer.add_scalar('Accuracy/Train_Balanced', train_metrics['balanced_accuracy'], epoch)
        tb_writer.add_scalar('Accuracy/Val_Overall', val_metrics['overall_accuracy'], epoch)
        tb_writer.add_scalar('Accuracy/Val_Balanced', val_metrics['balanced_accuracy'], epoch)
        tb_writer.add_scalar('F1/Val_Macro', val_metrics['macro_f1'], epoch)
        tb_writer.add_scalar('F1/Val_Weighted', val_metrics['weighted_f1'], epoch)
        tb_writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 记录各类别指标
        for i, (recall, precision, f1) in enumerate(zip(val_metrics['class_recalls'], 
                                                      val_metrics['class_precisions'], 
                                                      val_metrics['class_f1s'])):
            tb_writer.add_scalar(f'ClassMetrics/Class_{i}_Recall', recall, epoch)
            tb_writer.add_scalar(f'ClassMetrics/Class_{i}_Precision', precision, epoch)
            tb_writer.add_scalar(f'ClassMetrics/Class_{i}_F1', f1, epoch)
        
        # 每5个epoch记录模型参数分布（避免过度占用存储空间）
        if (epoch + 1) % 5 == 0:
            try:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # 记录参数值分布
                        tb_writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                        # 记录梯度分布  
                        tb_writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)
                        # 记录参数范数
                        tb_writer.add_scalar(f'ParamNorms/{name}', param.data.norm().item(), epoch)
                        tb_writer.add_scalar(f'GradNorms/{name}', param.grad.data.norm().item(), epoch)
            except Exception as e:
                logger.warning(f"记录参数分布时出现错误: {e}")
        
        # 获取内存使用情况
        memory_usage = get_memory_usage()
        logger.info(f'Epoch [{epoch+1}/{config.EPOCHS}]')
        logger.info(f'  Train - Loss: {train_loss:.4f}, Overall Acc: {train_metrics["overall_accuracy"]:.2f}%, Balanced Acc: {train_metrics["balanced_accuracy"]:.2f}%')
        logger.info(f'  Val   - Loss: {val_loss:.4f}, Overall Acc: {val_metrics["overall_accuracy"]:.2f}%, Balanced Acc: {val_metrics["balanced_accuracy"]:.2f}%')
        logger.info(f'  Val F1 - Macro: {val_metrics["macro_f1"]:.2f}%, Weighted: {val_metrics["weighted_f1"]:.2f}%')
        logger.info(f'  Class Recalls: {[f"{r:.1f}" for r in val_metrics["class_recalls"]]}%')
        logger.info(f'  Class F1s: {[f"{f:.1f}" for f in val_metrics["class_f1s"]]}%')
        logger.info(f'  Imbalance Ratio: {val_metrics["imbalance_ratio"]:.2f}')
        logger.info(f'  内存使用 - RAM: {memory_usage["ram"]}, GPU: {memory_usage["gpu"]}')
        
        # 强制刷新日志缓冲区，确保实时写入
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        
        # 保存最佳模型 - 使用平衡准确率作为评判标准
        val_balanced_acc = val_metrics['balanced_accuracy']
        if val_balanced_acc > best_accuracy:
            best_accuracy = val_balanced_acc
            best_model_path = os.path.join(config.MODEL_OUTPUT_DIR, 'vis_mfn_best.pth')
            save_checkpoint(model, optimizer, epoch, val_balanced_acc, best_accuracy, model_config, best_model_path)
            logger.info(f'🎉 新的最佳模型已保存 (平衡准确率: {val_balanced_acc:.2f}%)')
        
        # 定期保存检查点
        if (epoch + 1) % 3 == 0:
            checkpoint_path = os.path.join(config.MODEL_OUTPUT_DIR, f'vis_mfn_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, val_balanced_acc, best_accuracy, model_config, checkpoint_path)
    
    # 保存最终模型
    final_model_path = os.path.join(config.MODEL_OUTPUT_DIR, 'vis_mfn_final.pth')
    save_checkpoint(model, optimizer, config.EPOCHS-1, val_balanced_acc, best_accuracy, model_config, final_model_path)
    
    # 关闭 TensorBoard writer
    tb_writer.close()
    logger.info("训练完成！")
    logger.info(f"最佳验证准确率: {best_accuracy:.2f}%")

if __name__ == '__main__':
    main()