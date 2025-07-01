import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout

class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=5, dropout=0.5):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()  # 定义图卷积层列表
        self.bns = torch.nn.ModuleList()    # 定义批量归一化层列表
        
        # 初始化图卷积层和批量归一化层
        for i in range(num_layers):
            nn = Sequential(
                Linear(num_features if i == 0 else hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU()
            )
            self.convs.append(GINConv(nn))  # 添加 GIN 卷积层
            self.bns.append(BatchNorm1d(hidden_dim))  # 添加批量归一化层
        
        self.fc = Linear(hidden_dim, num_classes)  # 定义全连接层
        self.dropout = Dropout(dropout)  # 定义 Dropout 层

    def forward(self, x, edge_index, batch):
        """
        前向传播函数
        :param x: 节点特征
        :param edge_index: 边索引
        :param batch: 图的批次信息
        :return: 分类结果
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)  # 图卷积操作
            x = bn(x)  # 批量归一化
            x = F.relu(x)  # 激活函数
        x = global_add_pool(x, batch)  # 全局加和池化
        x = self.dropout(x)  # 应用 Dropout
        out = self.fc(x)  # 输出分类结果
        return out

    def extract_features(self, x, edge_index, batch):
        """
        特征提取函数
        :param x: 节点特征
        :param edge_index: 边索引
        :param batch: 图的批次信息
        :return: 聚合后的节点特征
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return global_add_pool(x, batch)  # 返回池化后的特征