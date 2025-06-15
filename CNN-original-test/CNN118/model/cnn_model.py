import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Model_118(nn.Module):
    def __init__(self, nfeat_node, nfeat_edge, nhid, n_classes, dropout=0.3):
        super(CNN_Model_118, self).__init__()

        # 对节点特征的卷积层
        self.conv1_node = nn.Conv1d(in_channels=1, out_channels=nhid, kernel_size=3, padding=1)
        self.conv2_node = nn.Conv1d(in_channels=nhid, out_channels=nhid, kernel_size=3, padding=1)

        # 对边特征的卷积层
        self.conv1_edge = nn.Conv1d(in_channels=1, out_channels=nhid, kernel_size=3, padding=1)
        self.conv2_edge = nn.Conv1d(in_channels=nhid, out_channels=nhid, kernel_size=3, padding=1)

        # 根据卷积后的特征图大小计算全连接层的输入维度
        node_out_features = nhid * (nfeat_node // 4)
        edge_out_features = nhid * (nfeat_edge // 4)
        combined_features = node_out_features + edge_out_features

        # 全连接层
        self.fc1 = nn.Linear(combined_features, nhid)
        self.fc2 = nn.Linear(nhid, n_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_nodes, x_edges, G_in=None, G_out=None):
        # 节点特征卷积
        x_nodes = x_nodes.unsqueeze(1)  # 添加通道维度 -> [batch_size, 1, 474]
        x_nodes = F.relu(self.conv1_node(x_nodes))  # -> [batch_size, nhid, 474]
        x_nodes = F.max_pool1d(x_nodes, kernel_size=2)  # -> [batch_size, nhid, 237]
        x_nodes = F.relu(self.conv2_node(x_nodes))  # -> [batch_size, nhid, 237]
        x_nodes = F.max_pool1d(x_nodes, kernel_size=2)  # -> [batch_size, nhid, 118]
        x_nodes = x_nodes.view(x_nodes.size(0), -1)  # 展平 -> [batch_size, nhid * 118]

        # 边特征卷积
        x_edges = x_edges.unsqueeze(1)  # 添加通道维度 -> [batch_size, 1, 747]
        x_edges = F.relu(self.conv1_edge(x_edges))  # -> [batch_size, nhid, 747]
        x_edges = F.max_pool1d(x_edges, kernel_size=2)  # -> [batch_size, nhid, 374]
        x_edges = F.relu(self.conv2_edge(x_edges))  # -> [batch_size, nhid, 374]
        x_edges = F.max_pool1d(x_edges, kernel_size=2)  # -> [batch_size, nhid, 187]
        x_edges = x_edges.view(x_edges.size(0), -1)  # 展平 -> [batch_size, nhid * 187]

        # 合并节点和边特征
        x = torch.cat((x_nodes, x_edges), dim=1)  # -> [batch_size, nhid * (118 + 187)]

        # 全连接层
        x = F.relu(self.fc1(x))  # -> [batch_size, nhid]
        x = self.dropout(x)
        x = self.fc2(x)  # -> [batch_size, n_classes]

        return x
