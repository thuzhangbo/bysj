import torch
import torch.nn.functional as F
import yaml
import os
from data import get_loaders
from model import GIN
from utils import load_model, get_num_features, get_num_classes, get_device, setup_logger
from sklearn.metrics import accuracy_score, classification_report

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    logger = setup_logger(config['output']['log_dir'], "evaluate")
    device = get_device()
    logger.info(f"Using device: {device}")

    try:
        test_loader, _, target_data = get_loaders(config, split='test')
        num_features = get_num_features(target_data)
        num_classes = get_num_classes(config['dataset']['target_classes'])
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    model = GIN(num_features, num_classes,
                hidden_dim=config['model']['gin_hidden_dim'],
                num_layers=config['model']['gin_layers'],
                dropout=config['model']['dropout']).to(device)
    try:
        model = load_model(model, os.path.join(config['output']['model_dir'], 'adapted_gin.pth'), device)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(data.y.cpu().numpy())

    try:
        acc = accuracy_score(y_true, y_pred)
        logger.info(f"Overall Accuracy: {acc:.4f}")
        logger.info("\n" + classification_report(y_true, y_pred, labels=config['dataset']['target_classes']))
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")

if __name__ == '__main__':
    main()