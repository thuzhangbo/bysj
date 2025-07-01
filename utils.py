import os
import csv
import torch
import random
import numpy as np
import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, path, filename):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, filename))

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def get_num_features(dataset):
    return dataset[0].num_features

def get_num_classes(classes):
    return len(classes)

def setup_logger(log_dir, log_name):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    # 文件日志
    fh = logging.FileHandler(os.path.join(log_dir, f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    # 控制台日志
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    # 避免重复日志
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def plot_curves(metrics, save_dir):
    """
    绘制训练/验证损失和验证准确率曲线，并保存为图片。
    metrics: List[dict]，每个dict包含'epoch', 'train_loss', 'val_loss', 'val_acc'
    save_dir: 图片保存目录
    """
    if not metrics:
        return
    epochs = [m['epoch'] for m in metrics]
    train_loss = [m['train_loss'] for m in metrics]
    val_loss = [m['val_loss'] for m in metrics]
    val_acc = [m['val_acc'] for m in metrics]

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs, val_acc, label='Val Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, 'train_curves.png')
    plt.savefig(img_path)
    plt.close()
    print(f"[Plot] Training curves saved to {img_path}")


def save_metrics(metrics, save_path):
    if not metrics:
        return
    keys = metrics[0].keys()
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)


def get_num_features(data_list):
    """从数据列表中获取特征维度"""
    if data_list and len(data_list) > 0:
        sample = data_list[0]
        return sample.x.size(1) if sample.x is not None else 1
    return 1

def get_num_classes(class_list):
    """获取类别数"""
    return len(class_list)