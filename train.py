import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from datasets.vis_dataloader import get_dataloader
from datasets.feature_extraction import feature_extraction_block
from models.resnet.resnet_cbam import resnet50_cbam
from models.resnet.resnet import resnet50, resnet34, resnet18, JointModel
from models.wuhan.encoder import Encoder
from models.efficientnet.efficientnet import (
    efficientnet_b0, efficientnet_b1,
    efficientnet_b0_supcon, efficientnet_b1_supcon, efficientnet_b2_supcon)
from utils.utils import set_seed, setup_logging, get_memory_usage, normalize_feature_channels, create_optimizer, get_lr_scheduler
from utils.loss import create_loss_function, supcon_loss
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


def train_one_epoch(model, dataloader, criterion, optimizer, accumulation_steps, epoch, scaler, supcon=False):
    '''
    训练一个epoch，支持梯度累积和混合精度
    model: 训练的模型
    dataloader: 训练数据的dataloader
    criterion: 损失函数
    optimizer: 优化器
    accumulation_steps: 梯度累积步数
    epoch: 当前epoch数
    scaler: 混合精度训练的scaler
    supcon: 是否增加对比学习
    '''
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    if supcon:
        ce_criterion = criterion
        temperature = config.SUPCON_TEMPERATURE
        supcon_weight = config.SUPCON_WEIGHT
        ce_weight = config.CE_WEIGHT
        running_supcon_loss = 0.0
        running_ce_loss = 0.0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}' if epoch is not None else 'Training')
    for batch_idx, batch_data in enumerate(pbar):
        if batch_data is None:
            continue

        # batch_data包含(original_images, augmented_images, labels)
        original_images, augmented_images, labels = batch_data
        original_images = original_images.to(config.DEVICE)
        augmented_images = augmented_images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        if config.MODEL_TYPE == 'wuhan':
            batch_features = original_images # wuhan模型只用original_images
        else:
            batch_features, num_channels = feature_extraction_block(original_images, augmented_images)
            batch_features = normalize_feature_channels(batch_features, depth_ch=1)  # 各通道标准化

        # 计算loss
        if scaler is not None: # amp
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                if supcon: # 对比学习
                    h, z, logits = model(batch_features)  # SupCon 训练：模型输出 h, z, logits
                    supcon_loss_val = supcon_loss(z, labels, temperature)  # supcon loss
                    ce_loss_val = ce_criterion(logits, labels)  # ce loss
                    loss = supcon_weight * supcon_loss_val + ce_weight * ce_loss_val  # 联合loss
                    running_supcon_loss += supcon_loss_val.item() * accumulation_steps
                    running_ce_loss += ce_loss_val.item() * accumulation_steps
                else:
                    outputs = model(batch_features)
                    loss = criterion(outputs, labels)

                loss = loss / accumulation_steps
            scaler.scale(loss).backward()  # 自动处理梯度溢出
        else:
            if supcon: # 对比学习
                h, z, logits = model(batch_features)
                supcon_loss_val = supcon_loss(z, labels, temperature)
                ce_loss_val = ce_criterion(logits, labels)
                loss = supcon_weight * supcon_loss_val + ce_weight * ce_loss_val
                running_supcon_loss += supcon_loss_val.item() * accumulation_steps
                running_ce_loss += ce_loss_val.item() * accumulation_steps
            else:
                outputs = model(batch_features)
                loss = criterion(outputs, labels)

            loss = loss / accumulation_steps
            loss.backward()

        # 更新参数，更新前先进行梯度裁剪
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)  # 混合精度下需先 unscale_，否则剪裁会作用于缩放后的梯度，导致错误
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

        # 获取预测结果用于计算准确率
        if supcon:
            _, predicted = torch.max(logits, 1)
        else:
            _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if len(all_predictions) > 0:
            clipped_predictions = np.clip(np.array(all_predictions), 0, config.NUM_CLASSES - 1)  # 确保预测标签在合法范围内，避免sklearn警告
            current_balanced_acc = 100 * balanced_accuracy_score(all_labels, clipped_predictions)  # 平衡准确率（各类别召回率的平均值）
            current_overall_acc = 100 * np.mean(np.array(all_labels) == clipped_predictions)
        else:
            current_balanced_acc = 0.0
            current_overall_acc = 0.0

        # 更新进度条
        if supcon:
            current_supcon_loss = running_supcon_loss / (batch_idx + 1)
            current_ce_loss = running_ce_loss / (batch_idx + 1)
            pbar.set_postfix({'Loss': f'{current_loss:.4f}(CE:{current_ce_loss:.4f} + SupCon:{current_supcon_loss:.4f})', 'Overall_Acc': f'{current_overall_acc:.2f}%', 'Bal_Acc': f'{current_balanced_acc:.2f}%', 'Accum': f'{((batch_idx + 1) % accumulation_steps) + 1}/{accumulation_steps}'})
        else:
            pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Overall_Acc': f'{current_overall_acc:.2f}%', 'Bal_Acc': f'{current_balanced_acc:.2f}%', 'Accum': f'{((batch_idx + 1) % accumulation_steps) + 1}/{accumulation_steps}'})

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

    # 计算最终指标
    if len(all_predictions) > 0:
        clipped_predictions = np.clip(np.array(all_predictions), 0, config.NUM_CLASSES - 1)
        balanced_acc = 100 * balanced_accuracy_score(all_labels, clipped_predictions)
        overall_acc = 100 * np.mean(np.array(all_labels) == clipped_predictions)
    else:
        balanced_acc = 0.0
        overall_acc = 0.0

    # 返回训练结果
    train_metrics = {'overall_accuracy': overall_acc, 'balanced_accuracy': balanced_acc}

    # 如果是 SupCon 训练，添加详细损失信息
    if supcon:
        final_supcon_loss = running_supcon_loss / len(dataloader)
        final_ce_loss = running_ce_loss / len(dataloader)
        train_metrics.update({'supcon_loss': final_supcon_loss, 'ce_loss': final_ce_loss})

    avg_loss = running_loss / len(dataloader)

    return avg_loss, train_metrics


def validate(model, dataloader, criterion, supcon=False):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    if supcon:
        temperature = config.SUPCON_TEMPERATURE
        supcon_weight = config.SUPCON_WEIGHT
        ce_weight = config.CE_WEIGHT
        running_supcon_loss = 0.0
        running_ce_loss = 0.0

    with torch.no_grad():
        for batch_data in dataloader:
            if batch_data is None:
                continue

            original_images, augmented_images, labels = batch_data
            original_images = original_images.to(config.DEVICE)
            augmented_images = augmented_images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            if config.MODEL_TYPE == 'wuhan':
                batch_features = original_images
            else:
                batch_features, num_channels = feature_extraction_block(original_images, augmented_images)
                batch_features = normalize_feature_channels(batch_features, depth_ch=1) # 各通道标准化

            # 生成预测结果
            if supcon:
                h, z, logits = model(batch_features)
                supcon_loss_val = supcon_loss(z, labels, temperature)
                ce_loss_val = criterion(logits, labels)
                loss = supcon_weight * supcon_loss_val + ce_weight * ce_loss_val
                running_supcon_loss += supcon_loss_val.item()
                running_ce_loss += ce_loss_val.item()
                outputs = logits
            else:
                outputs = model(batch_features)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 生成输出
    avg_loss = running_loss / len(dataloader)
    metrics, _, _  = calculate_metrics(all_labels, all_predictions, config.NUM_CLASSES)

    if supcon:
        final_supcon_loss = running_supcon_loss / len(dataloader)
        final_ce_loss = running_ce_loss / len(dataloader)
        metrics.update({'supcon_loss': final_supcon_loss, 'ce_loss': final_ce_loss})

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
    parser.add_argument('--weighted_sampler', action='store_true', help='是否在dataloader中使用加权采样器')  # weighted_sampler/weighted_loss 二选一
    parser.add_argument('--weighted_loss', action='store_true', help='是否在损失函数中使用类别权重')  # focal alpha
    parser.add_argument('--early_stopping', action='store_true', help='是否启用早停')
    parser.add_argument('--loss_type', type=str, choices=['crossentropy', 'focal', 'dice_ce'], default='crossentropy')
    parser.add_argument('--optimizer', type=str, choices=['adamw', 'sgd'], default='sgd', help='选择优化器类型 (adamw/sgd)')
    parser.add_argument('--seed', type=int, default=6666)
    args = parser.parse_args()

    # 初始设置
    set_seed(args.seed)
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    logger = setup_logging(config.MODEL_OUTPUT_DIR)
    tb_writer = SummaryWriter(log_dir=os.path.join(config.MODEL_OUTPUT_DIR, 'tensorboard'))

    # 基本信息打印
    logger.info(f"随机种子: {args.seed}")
    logger.info(f"批次大小: {config.BATCH_SIZE}, 有效批次大小: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"加权策略: 采样器={args.weighted_sampler}, 类别权重={args.weighted_loss}")
    logger.info(f"损失函数: {args.loss_type}")
    logger.info(f"优化器类型: {args.optimizer}")
    logger.info(f"模型: {config.MODEL_TYPE}")

    # 加载数据
    train_loader, val_loader, train_labels = get_dataloader(config.TRAIN_DATA_ROOT, 
                                                            config.VAL_DATA_ROOT, 
                                                            config.USE_AUGMENTATION, 
                                                            weighted_sampler=args.weighted_sampler, 
                                                            train_resize_mode=config.TRAIN_RESIZE_MODE, 
                                                            val_resize_mode=config.VAL_RESIZE_MODE)

    # 创建模型，模型参数在这里修改
    if config.MODEL_TYPE == 'wuhan':
        model = Encoder(3, config.NUM_CLASSES, use_dropout=False)

    elif config.MODEL_TYPE == 'resnet18':
        model = resnet18(in_channels=11, use_se=True, use_dilation=True, dilation_rates=[1, 1, 1, 2], use_psa=False)
    
    elif config.MODEL_TYPE == 'resnet34':
        model = resnet34(in_channels=11, use_se=True, use_dilation=True, dilation_rates=[1, 1, 1, 2], use_psa=False)
    
    elif config.MODEL_TYPE == 'resnet50':
        model = resnet50(in_channels=11, use_se=True, use_dilation=True, dilation_rates=[1, 1, 1, 2], use_psa=False)

    elif config.MODEL_TYPE == 'resnet18_supcon':
        base_encoder = resnet18(in_channels=11, use_se=True, use_dilation=True, dilation_rates=[1, 1, 1, 2], use_psa=False)
        model = JointModel(base_encoder, projection_dim=128, num_classes=config.NUM_CLASSES)
    
    elif config.MODEL_TYPE == 'resnet34_supcon':
        base_encoder = resnet34(in_channels=11, use_se=True, use_dilation=True, dilation_rates=[1, 1, 1, 2], use_psa=False)
        model = JointModel(base_encoder, projection_dim=128, num_classes=config.NUM_CLASSES)
    
    elif config.MODEL_TYPE == 'resnet50_supcon':
        base_encoder = resnet50(in_channels=11, use_se=True, use_dilation=True, dilation_rates=[1, 1, 1, 2], use_psa=False)
        model = JointModel(base_encoder, projection_dim=128, num_classes=config.NUM_CLASSES)
    
    # EfficientNet系列
    elif config.MODEL_TYPE == 'efficientnet_b0':
        model = efficientnet_b0(in_channels=11, num_classes=config.NUM_CLASSES, pretrained=False)
    
    elif config.MODEL_TYPE == 'efficientnet_b1':
        model = efficientnet_b1(in_channels=11, num_classes=config.NUM_CLASSES, pretrained=False)
    
    elif config.MODEL_TYPE == 'efficientnet_b0_supcon':
        model = efficientnet_b0_supcon(in_channels=11, projection_dim=128, num_classes=config.NUM_CLASSES, pretrained=False)
    
    elif config.MODEL_TYPE == 'efficientnet_b1_supcon':
        model = efficientnet_b1_supcon(in_channels=11, projection_dim=128, num_classes=config.NUM_CLASSES, pretrained=False)
    
    elif config.MODEL_TYPE == 'efficientnet_b2_supcon':
        model = efficientnet_b2_supcon(in_channels=11, projection_dim=128, num_classes=config.NUM_CLASSES, pretrained=False)
    
    else:
        raise ValueError(f"不支持的模型类型: {config.MODEL_TYPE}")

    model = model.to(config.DEVICE)

    # 损失函数
    criterion = create_loss_function(train_labels,
                                     loss_type=args.loss_type,
                                     use_weights=args.weighted_loss,
                                     weight_mode=config.WEIGHT_MODE, # 计算权重方式
                                     weight_smoothing=False,
                                     smooth_factor=config.SMOOTH_FACTOR,
                                     label_smoothing=config.LABEL_SMOOTHING)

    # 优化器
    optimizer = create_optimizer(model, optimizer_type=args.optimizer)

    # 学习率变化策略
    scheduler = get_lr_scheduler(optimizer, warmup_epochs=config.WARMUP_EPOCHS, total_epochs=config.EPOCHS, eta_min=config.ETA_MIN)

    # 初始化早停
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, min_delta=config.EARLY_STOPPING_MIN_DELTA)

    # 混合精度训练
    try:
        scaler = GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu')
    except:
        try:
            scaler = GradScaler()
        except:
            scaler = None

    # 训练变量
    start_epoch = 0
    best_accuracy = 0.0  # 验证集

    # 恢复训练
    if args.resume and os.path.exists(args.resume):
        logger.info(f"从检查点恢复训练: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=config.DEVICE, weights_only=True)
        except:
            checkpoint = torch.load(args.resume, map_location=config.DEVICE, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        logger.info(f"恢复训练从第 {start_epoch} 轮开始，当前最佳准确率: {best_accuracy:.4f}")

    # 训练循环
    logger.info("开始训练...")
    use_supcon = ('supcon' in config.MODEL_TYPE)
    for epoch in range(start_epoch, config.EPOCHS):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, config.GRADIENT_ACCUMULATION_STEPS, epoch, scaler, supcon=use_supcon)
        val_loss, val_metrics = validate(model, val_loader, criterion, supcon=use_supcon)
        scheduler.step()
        # scheduler.step(val_loss) # ReduceLROnPlateau

        # TensorBoard结果记录
        if use_supcon and 'supcon_loss' in train_metrics:
            tb_writer.add_scalar('Loss/Train_SupCon', train_metrics['supcon_loss'], epoch)
            tb_writer.add_scalar('Loss/Train_CE', train_metrics['ce_loss'], epoch)

            if 'supcon_loss' in val_metrics:
                tb_writer.add_scalar('Loss/Val_SupCon', val_metrics['supcon_loss'], epoch)
                tb_writer.add_scalar('Loss/Val_CE', val_metrics['ce_loss'], epoch)

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

        # 训练损失记录
        if use_supcon and 'supcon_loss' in train_metrics:
            logger.info(f'  Train - Loss: {train_loss:.4f}(CE:{train_metrics["ce_loss"]:.4f} + SupCon:{train_metrics["supcon_loss"]:.4f})')
            logger.info(f'  Train - Overall Acc: {train_metrics["overall_accuracy"]:.2f}%, Balanced Acc: {train_metrics["balanced_accuracy"]:.2f}%')
        else:
            logger.info(f'  Train - Loss: {train_loss:.4f}, Overall Acc: {train_metrics["overall_accuracy"]:.2f}%, Balanced Acc: {train_metrics["balanced_accuracy"]:.2f}%')

        # 验证损失记录
        if use_supcon and 'supcon_loss' in val_metrics:
            logger.info(f'  Val   - Loss: {val_loss:.4f}(CE:{val_metrics["ce_loss"]:.4f} + SupCon:{val_metrics["supcon_loss"]:.4f})')
            logger.info(f'  Val   - Overall Acc: {val_metrics["overall_accuracy"]:.2f}%, Balanced Acc: {val_metrics["balanced_accuracy"]:.2f}%')
        else:
            logger.info(f'  Val   - Loss: {val_loss:.4f}, Overall Acc: {val_metrics["overall_accuracy"]:.2f}%, Balanced Acc: {val_metrics["balanced_accuracy"]:.2f}%')

        logger.info(f'  Val F1 - Macro: {val_metrics["macro_f1"]:.2f}%, Weighted: {val_metrics["weighted_f1"]:.2f}%')
        logger.info(f'  Class Recalls: {[f"{r:.1f}" for r in val_metrics["class_recalls"]]}%')
        logger.info(f'  Class F1s: {[f"{f:.1f}" for f in val_metrics["class_f1s"]]}%')
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
            # 使用少数类别的平均召回率作为早停指标，而不是整体平衡准确率
            minority_classes = [2, 3, 4]  # 类别2/3/4是少数类别
            minority_recalls = [val_metrics['class_recalls'][i] for i in minority_classes]
            minority_avg_recall = np.mean(minority_recalls)

            if early_stopping(minority_avg_recall):
                logger.info(f'早停触发！在第 {epoch+1} 轮停止训练')
                logger.info(f'少数类别平均召回率: {minority_avg_recall:.2f}%')
                logger.info(f'最佳验证平衡准确率: {best_accuracy:.2f}%')
                break

    # 保存最终模型
    final_model_path = os.path.join(config.MODEL_OUTPUT_DIR, 'vis_final.pth')
    save_checkpoint(model, optimizer, config.EPOCHS - 1, val_balanced_acc, best_accuracy, final_model_path)

    tb_writer.close()
    logger.info("训练完成！")
    logger.info(f"最佳验证准确率: {best_accuracy:.2f}%")


if __name__ == '__main__':
    # 模拟命令行参数
    # sys.argv = [
    #     'train.py',
    #     '--loss_type', 'focal',  # 改为focal loss
    #     '--weighted_sampler',
    #     '--weighted_loss', # --weighted_loss
    #     '--early_stopping'
    # ]

    # focal + weighted_loss + resnet50 + se + dilation + 余弦退火
    # sys.argv = [
    #     'train.py',
    #     '--loss_type', 'focal',  # 改为focal loss
    #     '--early_stopping'
    # ]

    # crossentropy + weighted_loss + resnet34 + se + dilation + 余弦退火 + 26通道
    # sys.argv = [
    #     'train.py',
    #     '--loss_type', 'dice_ce',  # 改为focal loss
    #     '--weighted_loss',
    #     '--early_stopping'
    # ]

    # 使用SGD优化器的示例
    sys.argv = [
        'train.py',
        '--weighted_loss',
        '--early_stopping',
        '--loss_type', 'crossentropy',
        '--optimizer', 'sgd',
    ]

    main()
