import os
import copy
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from typing import Tuple, Dict, Any
from data import create_data_loaders, load_config
from model import GIN
from utils import set_seed, get_device, setup_logger, plot_curves, save_metrics

def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """评估模型性能"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    acc = correct / total if total > 0 else 0
    return acc, avg_loss

def train_one_epoch(model: torch.nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    return avg_loss

def get_model_features_and_classes(data_splits: Dict, source_classes: list) -> Tuple[int, int]:
    """获取模型输入特征数和类别数"""
    if data_splits['train_data']:
        sample_data = data_splits['train_data'][0]
        num_features = sample_data.x.size(1) if sample_data.x is not None else 1
    else:
        raise ValueError("No training data available to determine feature dimensions")
    
    num_classes = len(source_classes)
    return num_features, num_classes

def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                          train_loss: float, val_acc: float, save_dir: str, source_label_map: Dict,
                          config: Dict) -> None:
    """保存模型状态"""
    checkpoint_path = os.path.join(save_dir, f'latest_source_{config["model"]["name"]}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_acc': val_acc,
        'source_label_map': source_label_map,
        'config': config
    }, checkpoint_path)

def main():
    # === 配置与日志 ===
    config = load_config('config.yaml')
    set_seed(config['train']['seed'])
    
    # === 动态生成保存目录 ===
    model_name = config['model']['name']
    lr_str = str(config['train']['lr'])
    dataset_name = config['dataset']['name']
    dataset_dir = os.path.join(config['output']['model_dir'], dataset_name)
    save_dir = os.path.join(dataset_dir, f"{model_name}_{lr_str}")
    config['output']['model_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'best'), exist_ok=True)
    
    logger = setup_logger(config['output']['log_dir'], "train_source")
    device = get_device()
    logger.info(f"Using device: {device}")

    # === 数据加载 ===
    try:
        data_info = create_data_loaders(config, logger)
        source_loaders = data_info['source_loaders']
        source_splits = data_info['source_splits']
        source_label_map = data_info['source_label_map']
        
        train_loader = source_loaders['train_loader']
        val_loader = source_loaders['val_loader']
        
        if not train_loader:
            raise ValueError("No source training data available")
        if not val_loader:
            logger.warning("No source validation data available")
        
        num_features, num_classes = get_model_features_and_classes(source_splits, config['dataset']['source_classes'])
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # === 模型初始化 ===
    try:
        model = GIN(
            num_features, 
            num_classes,
            hidden_dim=config['model']['gin_hidden_dim'],
            num_layers=config['model']['gin_layers'],
            dropout=config['model']['dropout']
        ).to(device)

        optimizer = Adam(
            model.parameters(), 
            lr=float(config['train']['lr']), 
            weight_decay=float(config['train']['weight_decay'])
        )
        
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return

    # === 训练准备 ===
    best_acc = 0.0
    best_model_state = None
    best_epoch = -1
    early_stop_patience = config['train'].get('early_stop_patience', 15)
    no_improve_epochs = 0
    metrics = []

    # 检查是否存在 record.txt
    best_directory = os.path.join(dataset_dir, 'best')
    record_file_path = os.path.join(best_directory, 'record.txt')
    previous_best_acc = None

    if os.path.exists(record_file_path):
        with open(record_file_path, 'r') as record_file:
            previous_best_acc = float(record_file.readline().strip())
            logger.info(f"Previous best validation accuracy read from record.txt: {previous_best_acc:.4f}")

    # === 训练循环 ===
    logger.info("Starting training...")
    
    for epoch in range(1, config['train']['epochs'] + 1):
        try:
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_acc, val_loss = evaluate(model, val_loader, device) if val_loader else (0.0, 0.0)

            metrics.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc
            })

            logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # 每10个epoch绘图一次
            if epoch % 10 == 0 or epoch == 1:  
                try:
                    plot_curves(metrics, config['output']['model_dir'])
                except Exception as e:
                    logger.warning(f"Error plotting curves: {e}")

            save_model_checkpoint(model, optimizer, train_loss, val_acc, save_dir, source_label_map, config)

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_model_state = {
                    'model_state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_acc': val_acc,
                    'source_label_map': source_label_map,
                    'config': config
                }
                no_improve_epochs = 0
                logger.info(f"New best model at epoch {epoch} (Val Acc: {val_acc:.4f})")
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch} "
                            f"(no improvement in {early_stop_patience} epochs).")
                break

        except Exception as e:
            logger.error(f"Error in epoch {epoch}: {e}")
            continue

    # === 保存最优模型 ===
    if best_model_state is not None:
        best_model_path = os.path.join(best_directory, 'best_source_gin.pth')
        torch.save(best_model_state, best_model_path)
        logger.info(f"Best model saved to {best_model_path} "
                    f"(Val Acc: {best_acc:.4f} at epoch {best_epoch})")

        # 更新 record.txt
        if previous_best_acc is None or best_acc > previous_best_acc:
            with open(record_file_path, 'w') as record_file:
                record_file.write(f"{best_acc:.4f}\n")
                record_file.write(str(config) + '\n')  # 更新训练相关配置
            logger.info(f"Updated record.txt with new best validation accuracy: {best_acc:.4f}")

    else:
        logger.warning("No best model to save.")

    # === 保存训练曲线和指标 ===
    try:
        metrics_path = os.path.join(config['output']['model_dir'], 'train_metrics.csv')
        save_metrics(metrics, metrics_path)
        logger.info(f"Training/validation metrics saved to {metrics_path}")
        
        plot_curves(metrics, config['output']['model_dir'])
        logger.info("Final training curves saved")
        
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")

    # === 训练总结 ===
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_acc:.4f}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Total epochs trained: {len(metrics)}")
    
    return {
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'total_epochs': len(metrics),
        'metrics': metrics
    }

if __name__ == '__main__':
    main()