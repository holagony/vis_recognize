import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from datasets.vis_dataloader import get_dataloader
from datasets.feature_extraction import feature_extraction_block
from models.vismfn.model import VisMFN
from models.resnet.resnet_cbam import resnet50_cbam
from utils.utils import set_seed, setup_logging, get_memory_usage
from utils.loss import create_loss_function
from utils.metric import calculate_metrics
from utils import config

try:
   from torch import GradScaler # torch >= 2.3
except:
   from torch.cuda.amp import GradScaler

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        
    def __call__(self, val_score):
        if self.best_score is None or val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, eta_min):
    '''
    warmup + 余弦退火
    '''
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


def train_one_epoch(model, dataloader, criterion, optimizer, accumulation_steps=1, epoch=None, scaler=None):
    '''
    训练一个epoch，
    支持梯度累积和混合精度
    '''
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}' if epoch is not None else 'Training')
    for batch_idx, batch_data in enumerate(pbar):
        if batch_data is None:
            continue

        # batch_data包含(original_images, augmented_images, labels)
        original_images, augmented_images, labels = batch_data
        original_images = original_images.to(config.DEVICE)
        augmented_images = augmented_images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        batch_features, num_channels = feature_extraction_block(original_images, augmented_images)
        
        # 对融合后的特征进行标准化
        # 前3个通道使用ImageNet标准化参数
        rgb_mean = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]
        
        # 其余通道使用默认参数
        other_mean = [0.0] * (num_channels - 3)
        other_std = [1.0] * (num_channels - 3)
        
        # 标准化
        mean = rgb_mean + other_mean
        std = rgb_std + other_std
        mean_tensor = torch.tensor(mean, device=batch_features.device).view(-1, 1, 1)
        std_tensor = torch.tensor(std, device=batch_features.device).view(-1, 1, 1)
        batch_features = (batch_features - mean_tensor) / std_tensor

        # 计算loss
        if scaler is not None:
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(batch_features)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward() # 自动处理梯度溢出
        else:
            outputs = model(batch_features)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()
        
        # 更新参数，更新前先进行梯度裁剪
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer) # 混合精度下需先 unscale_，否则剪裁会作用于缩放后的梯度，导致错误
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_NORM)
                optimizer.step()
            optimizer.zero_grad()
        
        # loss和指标
        running_loss += loss.item() * accumulation_steps  # 恢复原始损失值用于记录
        current_loss = running_loss / (batch_idx + 1)
        
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if len(all_predictions) > 0:
            clipped_predictions = np.clip(np.array(all_predictions), 0, config.NUM_CLASSES - 1) # 确保预测标签在合法范围内，避免sklearn警告
            current_balanced_acc = 100 * balanced_accuracy_score(all_labels, clipped_predictions) # 平衡准确率（各类别召回率的平均值）
            current_overall_acc = 100 * np.mean(np.array(all_labels) == clipped_predictions)
        else:
            current_balanced_acc = 0.0
            current_overall_acc = 0.0

        # 更新进度条 
        pbar.set_postfix({'Loss': f'{current_loss:.4f}',
                          'Overall_Acc': f'{current_overall_acc:.2f}%',
                          'Bal_Acc': f'{current_balanced_acc:.2f}%',
                          'Accum': f'{((batch_idx + 1) % accumulation_steps) + 1}/{accumulation_steps}'})
        
        # TODO: 
        # 按batch记录loss
        # if batch_idx % 100 == 0:  # 每 100 个 Batch 打印一次
        #     print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    
    # 处理最后一个不完整的累积批次
    if len(dataloader) % accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_NORM)
            optimizer.step()
        optimizer.zero_grad()
    
    # 生成输出
    avg_loss = running_loss / len(dataloader)
    metrics, _ = calculate_metrics(all_labels, all_predictions, config.NUM_CLASSES)
    
    return avg_loss, metrics


def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if batch_data is None:
                continue

            original_images, augmented_images, labels = batch_data
            original_images = original_images.to(config.DEVICE)
            augmented_images = augmented_images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            batch_features, num_channels = feature_extraction_block(original_images, augmented_images)
            
            # 对融合后的特征进行标准化
            # 前3个通道使用ImageNet标准化参数
            rgb_mean = [0.485, 0.456, 0.406]
            rgb_std = [0.229, 0.224, 0.225]
            
            # 其余通道使用默认参数
            other_mean = [0.0] * (num_channels - 3)
            other_std = [1.0] * (num_channels - 3)
            
            # 标准化
            mean = rgb_mean + other_mean
            std = rgb_std + other_std
            mean_tensor = torch.tensor(mean, device=batch_features.device).view(-1, 1, 1)
            std_tensor = torch.tensor(std, device=batch_features.device).view(-1, 1, 1)
            batch_features = (batch_features - mean_tensor) / std_tensor

            # 生成预测结果
            outputs = model(batch_features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 生成输出
    avg_loss = running_loss / len(dataloader)
    metrics, _ = calculate_metrics(all_labels, all_predictions, config.NUM_CLASSES)
    
    return avg_loss, metrics


def save_checkpoint(model, optimizer, epoch, accuracy, best_accuracy, save_path):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'best_accuracy': best_accuracy}, save_path)


def main():
    parser = argparse.ArgumentParser(description='Vis Training')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--loss_type', type=str, choices=['crossentropy', 'focal'], default='crossentropy')
    parser.add_argument('--weighted_sampler', action='store_true', help='是否使用加权采样器') # weighted_sampler/weighted_loss 最好二选一
    parser.add_argument('--weighted_loss', action='store_true', help='是否在损失函数中使用类别权重')
    parser.add_argument('--early_stopping', action='store_true', help='是否启用早停')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    # 初始设置
    set_seed(args.seed)
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    logger = setup_logging(config.MODEL_OUTPUT_DIR)
    tb_writer = SummaryWriter(log_dir=os.path.join(config.MODEL_OUTPUT_DIR, 'tensorboard'))

    logger.info(f"随机种子: {args.seed}")
    logger.info(f"批次大小: {config.BATCH_SIZE}, 有效批次大小: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"加权策略: 采样器={args.weighted_sampler}, 损失权重={args.weighted_loss}")

    # 加载数据
    train_loader, val_loader, train_labels = get_dataloader(config.TRAIN_DATA_ROOT, config.VAL_DATA_ROOT, config.USE_AUGMENTATION, weighted_sampler=args.weighted_sampler)

    # 创建模型
    if config.MODEL_TYPE == 'resnet':
        model = resnet50_cbam(pretrained=False, in_channels=26)
    elif config.MODEL_TYPE == 'vismfn':
        model_kwargs = config.get_model_kwargs()
        model = VisMFN(**model_kwargs)
    model = model.to(config.DEVICE)

    # loss
    criterion = create_loss_function(train_labels, 
                                     loss_type=args.loss_type, 
                                     use_weights=args.weighted_loss, 
                                     weight_mode=config.WEIGHT_MODE,
                                     smooth_factor=config.SMOOTH_FACTOR, 
                                     label_smoothing=config.LABEL_SMOOTHING)
    
    # 优化器参数
    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, betas=config.BETAS, eps=config.EPS)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, betas=config.BETAS, eps=config.EPS)
    
    # 学习率
    # scheduler = get_lr_scheduler(optimizer, config.WARMUP_EPOCHS, config.EPOCHS, config.ETA_MIN) # warmup + 余弦退火
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 混合精度训练
    try:
        scaler = GradScaler(device='cuda') if torch.cuda.is_available() else None
    except:
        try:
            scaler = GradScaler() if torch.cuda.is_available() else None
        except:
            scaler = None

    # 训练变量
    start_epoch = 0
    best_accuracy = 0.0 # 验证集
    
    # 恢复训练
    if args.resume and os.path.exists(args.resume):
        logger.info(f"从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        logger.info(f"恢复训练从第 {start_epoch} 轮开始，当前最佳准确率: {best_accuracy:.4f}")
    
    # 训练循环
    logger.info("开始训练...")
    for epoch in range(start_epoch, config.EPOCHS):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, config.GRADIENT_ACCUMULATION_STEPS, epoch, scaler)
        val_loss, val_metrics = validate(model, val_loader, criterion)
        scheduler.step() # update learning rate
        # scheduler.step(val_loss)
        
        # TensorBoard结果记录
        tb_writer.add_scalar('Loss/Train', train_loss, epoch)
        tb_writer.add_scalar('Loss/Validation', val_loss, epoch)
        tb_writer.add_scalar('Accuracy/Train_Overall', train_metrics['overall_accuracy'], epoch)
        tb_writer.add_scalar('Accuracy/Train_Balanced', train_metrics['balanced_accuracy'], epoch)
        tb_writer.add_scalar('Accuracy/Val_Overall', val_metrics['overall_accuracy'], epoch)
        tb_writer.add_scalar('Accuracy/Val_Balanced', val_metrics['balanced_accuracy'], epoch)
        tb_writer.add_scalar('F1/Val_Macro', val_metrics['macro_f1'], epoch)
        tb_writer.add_scalar('F1/Val_Weighted', val_metrics['weighted_f1'], epoch)
        tb_writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        for i, (recall, precision, f1) in enumerate(zip(val_metrics['class_recalls'], val_metrics['class_precisions'], val_metrics['class_f1s'])):
            tb_writer.add_scalar(f'ClassMetrics/Class_{i}_Recall', recall, epoch)
            tb_writer.add_scalar(f'ClassMetrics/Class_{i}_Precision', precision, epoch)
            tb_writer.add_scalar(f'ClassMetrics/Class_{i}_F1', f1, epoch)
        
        # 日志记录
        memory_usage = get_memory_usage()
        logger.info(f'Epoch [{epoch+1}/{config.EPOCHS}]')
        logger.info(f'  Train - Loss: {train_loss:.4f}, Overall Acc: {train_metrics["overall_accuracy"]:.2f}%, Balanced Acc: {train_metrics["balanced_accuracy"]:.2f}%')
        logger.info(f'  Val   - Loss: {val_loss:.4f}, Overall Acc: {val_metrics["overall_accuracy"]:.2f}%, Balanced Acc: {val_metrics["balanced_accuracy"]:.2f}%')
        logger.info(f'  Val F1 - Macro: {val_metrics["macro_f1"]:.2f}%, Weighted: {val_metrics["weighted_f1"]:.2f}%')
        logger.info(f'  Class Recalls: {[f"{r:.1f}" for r in val_metrics["class_recalls"]]}%')
        logger.info(f'  Class F1s: {[f"{f:.1f}" for f in val_metrics["class_f1s"]]}%')
        logger.info(f'  Imbalance Ratio: {val_metrics["imbalance_ratio"]:.2f}')
        logger.info(f'  内存使用 - RAM: {memory_usage["ram"]}, GPU: {memory_usage["gpu"]}')
        
        # 保存模型
        # 最佳模型
        val_balanced_acc = val_metrics['balanced_accuracy']  # 使用平衡准确率（各类别召回率的平均值）作为评判标准
        if val_balanced_acc > best_accuracy:
            best_accuracy = val_balanced_acc
            best_model_path = os.path.join(config.MODEL_OUTPUT_DIR, 'vis_best.pth')
            save_checkpoint(model, optimizer, epoch, val_balanced_acc, best_accuracy, best_model_path)
            logger.info(f'新的最佳模型已保存 (平衡准确率: {val_balanced_acc:.2f}%)')
        
        # 每x轮保存一次
        if (epoch + 1) % 3 == 0:
            checkpoint_path = os.path.join(config.MODEL_OUTPUT_DIR, f'vis_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, val_balanced_acc, best_accuracy, checkpoint_path)

        # 早停
        if args.early_stopping:
            early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, min_delta=config.EARLY_STOPPING_MIN_DELTA)
            if early_stopping(val_balanced_acc):
                logger.info(f'早停触发！在第 {epoch+1} 轮停止训练')
                logger.info(f'最佳验证准确率: {best_accuracy:.2f}%')
                break

    # 保存最终模型
    final_model_path = os.path.join(config.MODEL_OUTPUT_DIR, 'vis_final.pth')
    save_checkpoint(model, optimizer, config.EPOCHS-1, val_balanced_acc, best_accuracy, final_model_path)
    
    tb_writer.close()
    logger.info("训练完成！")
    logger.info(f"最佳验证准确率: {best_accuracy:.2f}%")


if __name__ == '__main__':
    main()
