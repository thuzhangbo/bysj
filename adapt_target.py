import torch
import torch.nn.functional as F
from torch.optim import Adam
import yaml
import os
from data import get_loaders
from model import GIN
from utils import (
    set_seed, get_num_features, get_num_classes, get_device, setup_logger, plot_curves, save_metrics, load_model
)
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import copy

def kd_loss(student_logits, teacher_logits, T=1.0):
    """
    经典知识蒸馏损失（KL散度），T为温度
    """
    student_log_prob = F.log_softmax(student_logits / T, dim=1)
    teacher_prob = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (T * T)

def evaluate(model, data_loader, device, known_classes):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    known_correct = 0
    known_total = 0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
            total_loss += loss.item()
            # 统计已知类别准确率
            mask = torch.tensor([int(label) in known_classes for label in data.y.cpu()])
            if mask.sum() > 0:
                known_correct += (pred[mask] == data.y[mask]).sum().item()
                known_total += mask.sum().item()
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    acc = correct / total if total > 0 else 0
    known_acc = known_correct / known_total if known_total > 0 else 0
    return acc, avg_loss, known_acc

def main():
    # === 配置与日志 ===
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    set_seed(config['train']['seed'])
    model_name = config['model']['name']
    dataset_name = config['dataset']['name']
    lr = config['train']['lr']
    save_dir = os.path.join(config['output']['model_dir'], f"{model_name}_adapt_{lr}_{dataset_name}")
    config['output']['model_dir'] = save_dir
    logger = setup_logger(config['output']['log_dir'], "adapt_target")
    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Model/checkpoints will be saved to: {save_dir}")

    # === 加载目标域数据 ===
    try:
        _, _, target_data = get_loaders(config, split='test')
        num_features = get_num_features(target_data)
        num_classes = get_num_classes(config['dataset']['target_classes'])
        source_classes = config['dataset']['source_classes']
        logger.info(f"Loaded {len(target_data)} target samples.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    train_data, val_data = train_test_split(
        target_data, test_size=0.2, random_state=config['train']['seed'],
        stratify=[int(d.y) for d in target_data]
    )
    train_loader = DataLoader(train_data, batch_size=config['dataset']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['dataset']['batch_size'], shuffle=False)
    logger.info(f"Target Train/Val split: {len(train_data)}/{len(val_data)}")

    # === 加载教师模型 ===
    teacher = GIN(num_features, len(source_classes), 
                  hidden_dim=config['model']['gin_hidden_dim'],
                  num_layers=config['model']['gin_layers'],
                  dropout=config['model']['dropout']).to(device)
    teacher_path = os.path.join(config['output']['model_dir'].replace("_adapt_", "_", 1), 'best_source_gin.pth')
    if not os.path.exists(teacher_path):
        logger.error(f"Teacher model not found at {teacher_path}")
        return
    load_model(teacher, teacher_path, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # === 初始化学生模型 ===
    student = GIN(num_features, num_classes,
                  hidden_dim=config['model']['gin_hidden_dim'],
                  num_layers=config['model']['gin_layers'],
                  dropout=config['model']['dropout']).to(device)
    optimizer = Adam(student.parameters(), lr=float(config['train']['lr']),
                     weight_decay=float(config['train']['weight_decay']))

    best_acc = 0.0
    best_model_state = None
    best_epoch = -1
    early_stop_patience = 10
    no_improve_epochs = 0
    metrics = []

    # === 训练循环 ===
    for epoch in range(1, config['adapt']['epochs'] + 1):
        student.train()
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            # 学生输出
            student_out = student(data.x, data.edge_index, data.batch)
            # 教师输出（只对已知类别计算KD损失）
            with torch.no_grad():
                # 只保留已知类别的样本
                mask = torch.tensor([int(label) in source_classes for label in data.y.cpu()]).to(device)
                if mask.sum() > 0:
                    teacher_out = teacher(data.x[mask], data.edge_index, data.batch[mask])
                    student_known_out = student_out[mask][:, :len(source_classes)]
                    teacher_loss = kd_loss(student_known_out, teacher_out, T=1.0)
                else:
                    teacher_loss = 0.0
            # 交叉熵损失（全类别）
            ce_loss = F.cross_entropy(student_out, data.y)
            loss = ce_loss + teacher_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0

        val_acc, val_loss, known_acc = evaluate(student, val_loader, device, source_classes)
        logger.info(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Known Acc: {known_acc:.4f}")
        metrics.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'known_acc': known_acc
        })
        # 实时绘图
        if epoch % 1 == 0:
            plot_curves(metrics, config['output']['model_dir'])
        # 保存最新模型
        try:
            latest_model_path = os.path.join(config['output']['model_dir'], 'latest_student_gin.pth')
            os.makedirs(config['output']['model_dir'], exist_ok=True)
            torch.save(student.state_dict(), latest_model_path)
        except Exception as e:
            logger.error(f"Error saving latest student model: {e}")
        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(student.state_dict())
            no_improve_epochs = 0
            logger.info(f"New best student model at epoch {epoch} (Val Acc: {val_acc:.4f})")
        else:
            no_improve_epochs += 1
        # 早停
        if no_improve_epochs >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch} (no improvement in {early_stop_patience} epochs).")
            break

    # === 保存最优模型 ===
    if best_model_state is not None:
        try:
            best_model_path = os.path.join(config['output']['model_dir'], 'best_student_gin.pth')
            os.makedirs(config['output']['model_dir'], exist_ok=True)
            torch.save(best_model_state, best_model_path)
            logger.info(f"Best student model saved to {best_model_path} (Val Acc: {best_acc:.4f} at epoch {best_epoch})")
        except Exception as e:
            logger.error(f"Error saving best student model: {e}")
    else:
        logger.warning("No best student model to save.")

    # === 保存训练曲线 ===
    try:
        metrics_path = os.path.join(config['output']['model_dir'], 'adapt_metrics.csv')
        save_metrics(metrics, metrics_path)
        logger.info(f"Adaptation metrics saved to {metrics_path}")
    except Exception as e:
        logger.error(f"Error saving adaptation metrics: {e}")

if __name__ == '__main__':
    main()