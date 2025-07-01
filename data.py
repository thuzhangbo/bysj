from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import logging
import os
from collections import Counter, defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import random
import torch


def setup_logger(log_dir, log_name='data_loader'):
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    log_file = os.path.join(log_dir, f"{log_name}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def set_seed(seed=42):
    """设置随机种子以保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_tudataset(config, logger=None):
    """统一的数据集加载函数"""
    try:
        dataset = TUDataset(
            root=config['dataset']['data_root'], 
            name=config['dataset']['name']
        )
        if logger:
            logger.info(f"Successfully loaded dataset {config['dataset']['name']} "
                       f"from {config['dataset']['data_root']}, total graphs: {len(dataset)}")
        return dataset
    except Exception as e:
        error_msg = f"Failed to load TUDataset: {e}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)


def create_label_mapping(source_classes, target_classes):
    """
    创建全局标签映射，确保标签连续
    
    Args:
        source_classes: 源域类别列表
        target_classes: 目标域类别列表
    
    Returns:
        source_label_map: 源域标签映射 {原始标签: 新标签}
        target_label_map: 目标域标签映射 {原始标签: 新标签}
        unknown_label: 未知类别的新标签值
        overlap_classes: 重叠类别列表
        target_novel_classes: 目标域独有类别列表
    """
    source_classes = sorted(set(source_classes))
    target_classes = sorted(set(target_classes))
    
    # 找出重叠类别和目标域独有类别
    overlap_classes = sorted(set(source_classes) & set(target_classes))
    target_novel_classes = sorted(set(target_classes) - set(source_classes))
    
    # 源域标签映射：重叠类别映射为 0, 1, 2, ...
    source_label_map = {orig_label: new_label for new_label, orig_label in enumerate(overlap_classes)}
    
    # 目标域标签映射
    target_label_map = {}
    unknown_label = len(overlap_classes)  # 未知类别标签
    
    # 重叠类别使用相同的映射
    for orig_label in overlap_classes:
        target_label_map[orig_label] = source_label_map[orig_label]
    
    # 目标域独有类别统一映射为未知类别
    for orig_label in target_novel_classes:
        target_label_map[orig_label] = unknown_label
    
    return source_label_map, target_label_map, unknown_label, overlap_classes, target_novel_classes


def split_overlap_class_samples(dataset, source_classes, target_classes, 
                               source_ratio=0.7, random_state=42, logger=None):
    """
    将重叠类别的样本分配给源域和目标域
    
    Args:
        dataset: 完整数据集
        source_classes: 源域类别列表
        target_classes: 目标域类别列表  
        source_ratio: 源域在重叠类别中的样本比例
        random_state: 随机种子
        logger: 日志记录器
    
    Returns:
        source_data: 分配给源域的数据（只包含重叠类别）
        target_data: 分配给目标域的数据（包含重叠类别+独有类别）
    """
    source_set = set(source_classes)
    target_set = set(target_classes)
    overlap_classes = source_set & target_set
    target_novel_classes = target_set - source_set
    
    if logger:
        logger.info(f"Class allocation analysis:")
        logger.info(f"  Source classes: {sorted(source_set)}")
        logger.info(f"  Target classes: {sorted(target_set)}")
        logger.info(f"  Overlap classes (to be split): {sorted(overlap_classes)}")
        logger.info(f"  Target novel classes (unknown): {sorted(target_novel_classes)}")
    
    # 按类别分组所有相关数据
    class_to_data = defaultdict(list)
    for data in dataset:
        label = int(data.y)
        if label in source_set or label in target_set:
            class_to_data[label].append(data)
    
    source_data = []
    target_data = []
    
    # 处理重叠类别：按比例分配
    for cls in overlap_classes:
        class_samples = class_to_data[cls]
        if not class_samples:
            if logger:
                logger.warning(f"  Class {cls}: No samples found")
            continue
            
        # 按比例划分
        if len(class_samples) > 1:
            source_samples, target_samples = train_test_split(
                class_samples,
                train_size=source_ratio,
                random_state=random_state,
                shuffle=True
            )
        else:
            # 只有一个样本时，随机分配
            random.seed(random_state)
            if random.random() < source_ratio:
                source_samples, target_samples = class_samples, []
            else:
                source_samples, target_samples = [], class_samples
        
        source_data.extend(source_samples)
        target_data.extend(target_samples)
        
        if logger:
            logger.info(f"  Class {cls}: {len(class_samples)} total -> "
                       f"Source: {len(source_samples)}, Target: {len(target_samples)}")
    
    # 处理目标域独有类别（新类别）：全部分配给目标域
    for cls in target_novel_classes:
        class_samples = class_to_data[cls]
        target_data.extend(class_samples)
        if logger and class_samples:
            logger.info(f"  Class {cls} (novel/unknown): {len(class_samples)} samples -> Target")
    
    # 打乱数据顺序
    random.shuffle(source_data)
    random.shuffle(target_data)
    
    if logger:
        logger.info(f"Final allocation:")
        logger.info(f"  Source domain: {len(source_data)} samples")
        logger.info(f"  Target domain: {len(target_data)} samples")
    
    return source_data, target_data


def apply_label_mapping(data_list, label_map):
    """应用标签映射到数据列表"""
    for data in data_list:
        orig_label = int(data.y)
        if orig_label in label_map:
            data.y = label_map[orig_label]
        else:
            raise ValueError(f"Label {orig_label} not found in label mapping")


def stratified_split_train_val(data_list, val_size=0.2, random_state=42):
    """
    按类别进行分层划分数据集为训练集和验证集
    """
    if not data_list:
        return [], []
    
    # 按标签分组
    label_to_data = defaultdict(list)
    for data in data_list:
        label = int(data.y)
        label_to_data[label].append(data)
    
    train_data, val_data = [], []
    
    for label, class_data in label_to_data.items():
        if len(class_data) == 1:
            # 如果某类只有一个样本，放到训练集
            train_data.extend(class_data)
            continue
        
        # 划分训练集和验证集
        class_train, class_val = train_test_split(
            class_data, 
            test_size=val_size, 
            random_state=random_state,
            shuffle=True
        )
        
        train_data.extend(class_train)
        val_data.extend(class_val)
    
    # 打乱数据顺序
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    return train_data, val_data


def get_source_loaders(config, logger=None):
    """
    获取源域数据加载器（包含训练集和验证集）
    
    Returns:
        loaders: 包含train_loader, val_loader的字典
        data_splits: 包含train_data, val_data的字典
        source_label_map: 源域标签映射字典
        class_info: 类别信息字典
    """
    # 设置随机种子
    if 'random_seed' in config.get('training', {}):
        set_seed(config['training']['random_seed'])
    
    # 加载数据集
    dataset = load_tudataset(config, logger)
    
    source_classes = config['dataset']['source_classes']
    target_classes = config['dataset']['target_classes']
    batch_size = config['dataset']['batch_size']
    
    # 获取数据分配参数
    allocation_config = config['dataset'].get('allocation', {})
    source_ratio = allocation_config.get('source_ratio', 0.7)
    random_state = allocation_config.get('random_state', 42)
    
    # 创建标签映射
    source_label_map, target_label_map, unknown_label, overlap_classes, target_novel_classes = create_label_mapping(
        source_classes, target_classes
    )
    
    # 分配重叠类别的样本
    source_data, _ = split_overlap_class_samples(
        dataset, source_classes, target_classes,
        source_ratio=source_ratio, random_state=random_state, logger=logger
    )
    
    if not source_data:
        warning_msg = f"No source data found after allocation"
        if logger:
            logger.warning(warning_msg)
        raise ValueError(warning_msg)
    
    # 应用标签映射
    apply_label_mapping(source_data, source_label_map)
    
    # 获取划分参数
    split_config = config['dataset'].get('split', {})
    val_size = split_config.get('val_size', 0.2)
    split_random_state = split_config.get('random_state', 42)
    
    # 划分数据集
    train_data, val_data = stratified_split_train_val(
        source_data, val_size=val_size, random_state=split_random_state
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) if val_data else None
    
    # 记录信息
    if logger:
        train_dist = Counter([int(d.y) for d in train_data])
        val_dist = Counter([int(d.y) for d in val_data]) if val_data else {}
        
        logger.info(f"Source domain setup:")
        logger.info(f"  Original classes: {source_classes}")
        logger.info(f"  Overlap classes used: {sorted(overlap_classes)}")
        logger.info(f"  Label mapping: {source_label_map}")
        logger.info(f"  Total data after allocation: {len(source_data)}")
        logger.info(f"  Data split - Train: {len(train_data)}, Val: {len(val_data)}")
        logger.info(f"  Train label distribution: {dict(train_dist)}")
        if val_data:
            logger.info(f"  Val label distribution: {dict(val_dist)}")
    
    loaders = {
        'train_loader': train_loader,
        'val_loader': val_loader
    }
    
    data_splits = {
        'train_data': train_data,
        'val_data': val_data,
        'all_data': source_data
    }
    
    class_info = {
        'overlap_classes': overlap_classes,
        'target_novel_classes': target_novel_classes,
        'unknown_label': unknown_label
    }
    
    return loaders, data_splits, source_label_map, class_info


def get_target_loaders(config, source_label_map, class_info, logger=None):
    """
    获取目标域数据加载器（包含训练集和验证集）
    
    Args:
        config: 配置字典
        source_label_map: 源域标签映射
        class_info: 类别信息字典
        logger: 日志记录器
    
    Returns:
        loaders: 包含train_loader, val_loader的字典
        data_splits: 包含train_data, val_data的字典
        unknown_label: 未知类别标签值
        target_label_map: 目标域标签映射字典
    """
    # 设置随机种子
    if 'random_seed' in config.get('training', {}):
        set_seed(config['training']['random_seed'])
    
    # 加载数据集
    dataset = load_tudataset(config, logger)
    
    source_classes = config['dataset']['source_classes']
    target_classes = config['dataset']['target_classes']
    batch_size = config['dataset']['batch_size']
    
    # 获取数据分配参数
    allocation_config = config['dataset'].get('allocation', {})
    source_ratio = allocation_config.get('source_ratio', 0.7)
    random_state = allocation_config.get('random_state', 42)
    
    # 创建标签映射
    _, target_label_map, unknown_label, overlap_classes, target_novel_classes = create_label_mapping(
        source_classes, target_classes
    )
    
    # 分配样本（获取目标域部分）
    _, target_data = split_overlap_class_samples(
        dataset, source_classes, target_classes,
        source_ratio=source_ratio, random_state=random_state, logger=logger
    )
    
    if not target_data:
        warning_msg = f"No target data found after allocation"
        if logger:
            logger.warning(warning_msg)
        raise ValueError(warning_msg)
    
    # 应用标签映射
    apply_label_mapping(target_data, target_label_map)
    
    # 获取划分参数
    split_config = config['dataset'].get('split', {})
    val_size = split_config.get('val_size', 0.2)
    split_random_state = split_config.get('random_state', 42)
    
    # 划分目标域数据集
    train_data, val_data = stratified_split_train_val(
        target_data, val_size=val_size, random_state=split_random_state
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) if train_data else None
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) if val_data else None
    
    # 记录信息
    if logger:
        train_label_list = [int(d.y) for d in train_data] if train_data else []
        val_label_list = [int(d.y) for d in val_data] if val_data else []
        all_label_list = [int(d.y) for d in target_data]
        
        train_counter = Counter(train_label_list)
        val_counter = Counter(val_label_list)
        all_counter = Counter(all_label_list)
        
        # 统计已知和未知类别数量
        known_count = sum(count for label, count in all_counter.items() if label != unknown_label)
        unknown_count = all_counter.get(unknown_label, 0)
        
        logger.info(f"Target domain setup:")
        logger.info(f"  Original classes: {target_classes}")
        logger.info(f"  Overlap classes: {sorted(overlap_classes)}")
        logger.info(f"  Novel classes: {sorted(target_novel_classes)} -> mapped to unknown label {unknown_label}")
        logger.info(f"  Label mapping: {target_label_map}")
        logger.info(f"  Total data after allocation: {len(target_data)}")
        logger.info(f"  Data split - Train: {len(train_data)}, Val: {len(val_data)}")
        logger.info(f"  Total count: {len(target_data)} (known: {known_count}, unknown: {unknown_count})")
        logger.info(f"  Total label distribution: {dict(all_counter)}")
        if train_data:
            logger.info(f"  Train label distribution: {dict(train_counter)}")
        if val_data:
            logger.info(f"  Val label distribution: {dict(val_counter)}")
    
    loaders = {
        'train_loader': train_loader,
        'val_loader': val_loader
    }
    
    data_splits = {
        'train_data': train_data,
        'val_data': val_data,
        'all_data': target_data
    }
    
    return loaders, data_splits, unknown_label, target_label_map


def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    if config_path.endswith(('.yaml', '.yml')):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif config_path.endswith('.json'):
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError("仅支持 yaml/yml 或 json 格式的配置文件")


def create_data_loaders(config, logger=None):
    """
    创建完整的数据加载器集合
    
    Returns:
        data_info: 包含所有加载器和相关信息的字典
    """
    if logger:
        logger.info("开始创建数据加载器...")
    
    # 获取源域数据加载器
    source_loaders, source_splits, source_label_map, class_info = get_source_loaders(config, logger)
    
    # 获取目标域数据加载器
    target_loaders, target_splits, unknown_label, target_label_map = get_target_loaders(
        config, source_label_map, class_info, logger
    )
    
    data_info = {
        # 源域相关
        'source_loaders': source_loaders,
        'source_splits': source_splits,
        'source_label_map': source_label_map,
        
        # 目标域相关
        'target_loaders': target_loaders,
        'target_splits': target_splits,
        'target_label_map': target_label_map,
        'unknown_label': unknown_label,
        
        # 类别信息
        'overlap_classes': class_info['overlap_classes'],
        'target_novel_classes': class_info['target_novel_classes'],
        
        # 其他信息
        'num_source_classes': len(source_label_map),
        'num_target_classes': len(set(target_label_map.values())),
        'source_classes': config['dataset']['source_classes'],
        'target_classes': config['dataset']['target_classes'],
    }
    
    if logger:
        logger.info("数据加载器创建完成!")
        logger.info(f"源域类别数: {data_info['num_source_classes']} (重叠类别)")
        logger.info(f"目标域类别数: {data_info['num_target_classes']} ({len(class_info['overlap_classes'])}个重叠类别 + 1个未知类别)")
        logger.info(f"重叠类别: {sorted(data_info['overlap_classes'])}")
        logger.info(f"目标域新类别: {sorted(data_info['target_novel_classes'])}")
        
        # 验证数据分配的正确性
        verify_data_allocation(data_info, logger)
    
    return data_info


def verify_data_allocation(data_info, logger=None):
    """验证数据分配的正确性，确保源域和目标域没有重叠样本"""
    source_data = data_info['source_splits']['all_data']
    target_data = data_info['target_splits']['all_data']
    
    # 这里简单验证数据对象的身份
    source_ids = set(id(data) for data in source_data)
    target_ids = set(id(data) for data in target_data)
    
    overlap_count = len(source_ids & target_ids)
    
    if logger:
        logger.info(f"Data allocation verification:")
        logger.info(f"  Source data samples: {len(source_data)}")
        logger.info(f"  Target data samples: {len(target_data)}")
        logger.info(f"  Overlapping samples: {overlap_count}")
        
        if overlap_count == 0:
            logger.info("  ✓ Data allocation is correct - no overlapping samples")
        else:
            logger.warning(f"  ✗ Found {overlap_count} overlapping samples!")


def analyze_dataset_splits(data_info, logger=None):
    """分析数据集划分情况"""
    source_splits = data_info['source_splits']
    target_splits = data_info['target_splits']
    source_label_map = data_info['source_label_map']
    target_label_map = data_info['target_label_map']
    unknown_label = data_info['unknown_label']
    
    print("\n=== 数据集划分分析 ===")
    
    # 源域分析
    print(f"\n源域数据:")
    print(f"  配置类别: {sorted(data_info['source_classes'])}")
    print(f"  实际使用类别(重叠): {sorted(data_info['overlap_classes'])}")
    print(f"  映射后类别: {sorted(source_label_map.values())}")
    print(f"  标签映射: {source_label_map}")
    print(f"  总数据量: {len(source_splits['all_data'])}")
    print(f"  训练集大小: {len(source_splits['train_data'])}")
    print(f"  验证集大小: {len(source_splits['val_data'])}")
    
    if source_splits['train_data']:
        train_dist = Counter([int(d.y) for d in source_splits['train_data']])
        print(f"  训练集标签分布: {dict(train_dist)}")
    
    if source_splits['val_data']:
        val_dist = Counter([int(d.y) for d in source_splits['val_data']])
        print(f"  验证集标签分布: {dict(val_dist)}")
    
    # 目标域分析
    print(f"\n目标域数据:")
    print(f"  配置类别: {sorted(data_info['target_classes'])}")
    print(f"  重叠类别: {sorted(data_info['overlap_classes'])}")
    print(f"  新类别(未知): {sorted(data_info['target_novel_classes'])}")
    print(f"  映射后类别: {sorted(set(target_label_map.values()))}")
    print(f"  未知类别标签: {unknown_label}")
    print(f"  标签映射: {target_label_map}")
    print(f"  总数据量: {len(target_splits['all_data'])}")
    print(f"  训练集大小: {len(target_splits['train_data'])}")
    print(f"  验证集大小: {len(target_splits['val_data'])}")
    
    if target_splits['train_data']:
        train_dist = Counter([int(d.y) for d in target_splits['train_data']])
        print(f"  训练集标签分布: {dict(train_dist)}")
    
    if target_splits['val_data']:
        val_dist = Counter([int(d.y) for d in target_splits['val_data']])
        print(f"  验证集标签分布: {dict(val_dist)}")


def analyze_tudataset_from_config_file(config_path='./config.yaml'):
    """从配置文件分析TUDataset"""
    config = load_config(config_path)
    
    try:
        dataset = TUDataset(
            root=config['dataset']['data_root'], 
            name=config['dataset']['name']
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load TUDataset: {e}")
    
    print(f"数据集总图数: {len(dataset)}")
    
    # 统计基本信息
    all_labels = []
    num_nodes_list = []
    num_edges_list = []
    node_feature_dims = set()
    edge_feature_dims = set()
    all_degrees = []
    
    for data in dataset:
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        num_node_features = data.num_node_features
        num_edge_features = data.num_edge_features if data.edge_attr is not None else 0
        label = int(data.y)
        
        all_labels.append(label)
        num_nodes_list.append(num_nodes)
        num_edges_list.append(num_edges)
        node_feature_dims.add(num_node_features)
        edge_feature_dims.add(num_edge_features)
        
        if data.edge_index is not None and data.edge_index.size(1) > 0:
            degrees = np.bincount(data.edge_index[0].cpu().numpy(), minlength=num_nodes)
            all_degrees.extend(degrees)
    
    # 输出统计结果
    label_counter = Counter(all_labels)
    print(f"标签类别数: {len(label_counter)}")
    print(f"各类别样本数: {dict(label_counter)}")
    print(f"每个图的节点数: 平均 {np.mean(num_nodes_list):.2f}, "
          f"最大 {np.max(num_nodes_list)}, 最小 {np.min(num_nodes_list)}")
    print(f"每个图的边数: 平均 {np.mean(num_edges_list):.2f}, "
          f"最大 {np.max(num_edges_list)}, 最小 {np.min(num_edges_list)}")
    print(f"节点特征维度: {node_feature_dims}")
    print(f"边特征维度: {edge_feature_dims}")
    
    if all_degrees:
        print(f"节点度: 平均 {np.mean(all_degrees):.2f}, "
              f"最大 {np.max(all_degrees)}, 最小 {np.min(all_degrees)}")
    
    print(f"所有标签: {sorted(set(all_labels))}")


# 示例使用函数
def example_usage(config_path='config.yaml'):
    """完整使用示例"""
    # 设置日志
    logger = setup_logger('./logs', 'data_loader')
    
    # 加载配置
    config = load_config(config_path)
    
    try:
        # 创建所有数据加载器
        data_info = create_data_loaders(config, logger)
        
        # 分析数据集划分
        analyze_dataset_splits(data_info, logger)
        
        return data_info
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise


if __name__ == "__main__":
    # 分析数据集
    analyze_tudataset_from_config_file('config.yaml')
    
    # 运行完整示例
    # example_usage('config.yaml')