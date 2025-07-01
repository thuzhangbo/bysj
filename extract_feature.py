import torch
import yaml
import os
from data import get_loaders
from model import GIN
from utils import load_model, get_num_features, get_num_classes, get_device, setup_logger

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    logger = setup_logger(config['output']['log_dir'], "extract_feature")
    device = get_device()
    logger.info(f"Using device: {device}")

    try:
        _, _, target_data = get_loaders(config, split='test')
        num_features = get_num_features(target_data)
        num_classes = get_num_classes(config['dataset']['source_classes'])
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    model = GIN(num_features, num_classes,
                hidden_dim=config['model']['gin_hidden_dim'],
                num_layers=config['model']['gin_layers'],
                dropout=config['model']['dropout']).to(device)
    try:
        model = load_model(model, os.path.join(config['output']['model_dir'], 'source_gin.pth'), device)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for data in target_data:
            data = data.to(device)
            feat = model.extract_features(data.x, data.edge_index, data.batch)
            features.append(feat.cpu())
            labels.append(data.y.cpu())
    try:
        torch.save({'features': torch.cat(features), 'labels': torch.cat(labels)},
                   os.path.join(config['output']['model_dir'], 'target_features.pt'))
        logger.info(f"Features saved to {os.path.join(config['output']['model_dir'], 'target_features.pt')}")
    except Exception as e:
        logger.error(f"Error saving features: {e}")

if __name__ == '__main__':
    main()