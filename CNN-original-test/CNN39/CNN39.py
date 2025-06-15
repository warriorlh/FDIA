import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss, jaccard_score, classification_report, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pywt
from skmultilearn.model_selection import IterativeStratification  # 导入 IterativeStratification


from model.cnn_model import CNN_Model


# 设置随机种子以确保实验结果的可重复性
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 离散小波变换函数
def apply_dwt(data, wavelet='db1'):
    coeffs = pywt.wavedec(data, wavelet, mode='symmetric', axis=1)
    return np.hstack(coeffs)

# 加载数据
df = pd.read_csv("39节点.csv")

# 定义节点数
n_node = 39  # 39节点系统

# 提取节点特征：电压相角（39列）+ 电压幅值（39列） + 有功功率注入（39列） + 无功功率注入（39列）
X_nodes = np.concatenate((
    np.array(df.values)[:, 132:170],   # 电压相角
    np.array(df.values)[:, 302:340],   # 电压幅值
    np.array(df.values)[:, 93:131],    # 有功功率注入
    np.array(df.values)[:, 263:301]    # 无功功率注入
), axis=1)

# 提取边特征：有功功率流（92列，包括pf和pt）+ 无功功率流（92列，包括qf和qt）
X_edges = np.concatenate((
    np.array(df.values)[:, 1:46],      # 功率流起始测量值（pf）
    np.array(df.values)[:, 47:92],     # 功率流终点测量值（pt）
    np.array(df.values)[:, 171:216],   # 无功功率流起始测量值（qf）
    np.array(df.values)[:, 217:262]    # 无功功率流终点测量值（qt）
), axis=1)

# 构建多标签输出，341到343列表示可能的攻击位置，输出39维的多标签
y_labels = np.zeros((df.shape[0], n_node))

# 遍历每一列，找到所有非零的攻击节点并标记到对应位置
for col in [340, 341, 342]:  # 修正后的列索引
    for idx, node_idx in enumerate(df.values[:, col]):
        if node_idx != 0:
            y_labels[idx, int(node_idx) - 1] = 1  # 将相应节点的位置置为1

# 调试信息：检查每个节点的标签分布情况，确保正常和攻击数据都被包含
print("标签处理后，每个节点的标签分布情况：")
for i in range(n_node):
    print(f"Node {i+1}: {np.sum(y_labels[:, i])} instances")

# 调试信息：统计正常样本的数量（所有标签为 0 的行）
normal_samples = np.sum(np.all(y_labels == 0, axis=1))
print(f"正常样本的数量: {normal_samples}")

# 应用DWT到节点特征和边特征
X_nodes_dwt = apply_dwt(X_nodes)
X_edges_dwt = apply_dwt(X_edges)

# 调试信息：查看数据形状
print(f"X_nodes_dwt shape after DWT: {X_nodes_dwt.shape}")
print(f"X_edges_dwt shape after DWT: {X_edges_dwt.shape}")

# 归一化节点特征和边特征
scaler = StandardScaler()
X_nodes_dwt = scaler.fit_transform(X_nodes_dwt)
X_edges_dwt = scaler.fit_transform(X_edges_dwt)

# 创建有向图邻接矩阵
adj_in = np.zeros((n_node, n_node))
adj_out = np.zeros((n_node, n_node))

# IEEE 39节点系统的连接关系
connections = [
    (1, 2),  (1, 39),
    (2, 3),  (2, 25),
    (3, 4),  (3, 18),
    (4, 5),  (4, 14),
    (5, 6),  (6, 7),  (6, 11),
    (7, 8),
    (8, 9),
    (9, 39),
    (10, 11), (10, 13), (10, 32),
    (12, 11), (12, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17), (16, 19), (16, 21),
    (17, 18), (17, 27),
    (19, 20), (19, 33),
    (20, 34),
    (21, 22),
    (22, 23), (22, 35),
    (23, 24),
    (24, 25),
    (25, 26),
    (26, 27), (26, 28), (26, 29),
    (28, 29),
    (29, 38), (29, 30),
    (30, 31),
    (31, 32),
    (32, 33),
    (33, 37),
    (34, 35),
    (35, 36),
    (36, 37),
    (37, 38), (37, 39)
]

# 根据连接关系生成邻接矩阵
for (u, v) in connections:
    adj_in[v - 1, u - 1] = 1  # -1 是因为节点从1开始编号，但在数组中从0开始
    adj_out[u - 1, v - 1] = 1

# 确保邻接矩阵是3D张量
G_in = np.expand_dims(adj_in, axis=0)
G_out = np.expand_dims(adj_out, axis=0)

# 合并节点和边特征以便进行数据划分
X_combined = np.hstack((X_nodes_dwt, X_edges_dwt))

# 使用 IterativeStratification 进行多标签分层采样
stratifier = IterativeStratification(n_splits=2, order=1)
train_idx, test_idx = next(stratifier.split(X_combined, y_labels))

# 划分训练集和测试集
X_train_combined = X_combined[train_idx]
X_test_combined = X_combined[test_idx]
y_train = y_labels[train_idx]
y_test = y_labels[test_idx]

# 将合并后的特征再分开成节点特征和边特征
X_train_nodes_resampled = X_train_combined[:, :X_nodes_dwt.shape[1]]
X_train_edges_resampled = X_train_combined[:, X_nodes_dwt.shape[1]:]

X_test_nodes = X_test_combined[:, :X_nodes_dwt.shape[1]]
X_test_edges = X_test_combined[:, X_nodes_dwt.shape[1]:]


# 生成DataLoader
batch_size = 64  # 继续使用64的批量大小
train_dataloader = DataLoader(TensorDataset(torch.tensor(X_train_nodes_resampled).float(),
                              torch.tensor(X_train_edges_resampled).float(),
                              torch.tensor(y_train).float()), batch_size=batch_size, shuffle=True, num_workers=0)

test_dataloader = DataLoader(TensorDataset(torch.tensor(X_test_nodes).float(),
                             torch.tensor(X_test_edges).float(),
                             torch.tensor(y_test).float()), batch_size=batch_size, shuffle=False, num_workers=0)


# 迭代 DataLoader 并打印每个批次的输入形状
for X_nodes_batch, X_edges_batch, y_batch in train_dataloader:
    print(f"Batch X_nodes shape: {X_nodes_batch.shape}")
    print(f"Batch X_edges shape: {X_edges_batch.shape}")
    print(f"Batch y shape: {y_batch.shape}")
    break  # 只打印一个批次的数据


# 调整后的模型设置
epochs = 35  # 保持训练轮数不变
hidden_size = 256  # 使用256的隐藏层大小
learning_rate = 0.001  # 调整学习率为0.001



# model = CNN_Model(nfeat_node=X_nodes_dwt.shape[1], nfeat_edge=X_edges_dwt.shape[1], nhid=256, n_classes=39, dropout=0.3)



# 使用 Binary Cross-Entropy (BCE) 作为损失函数，适合多标签分类
loss = nn.BCEWithLogitsLoss()

# 使用带L2正则化的AdamW优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# 训练函数
def train(epochs, model, loss_func, optimizer, scheduler, train_loader, G_in, G_out):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        y_true = []
        y_pred_all = []
        for X_nodes, X_edges, y in train_loader:
            optimizer.zero_grad()

            y_pred = model(X_nodes, X_edges, G_in, G_out)

            loss_val = loss_func(y_pred, y)
            loss_val.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss_val.item() * y.size(0)
            y_true.append(y)
            y_pred_all.append(y_pred)

        epoch_loss /= len(train_loader.dataset)
        scheduler.step(epoch_loss)

        y_true = torch.cat(y_true, dim=0).detach().numpy()
        y_pred_all = torch.cat(y_pred_all, dim=0).detach().numpy()

        y_pred_labels = (y_pred_all > 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred_labels)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}")

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np


def evaluate(model, test_loader, G_in, G_out):
    model.eval()
    y_true = []
    y_pred_all = []
    with torch.no_grad():
        for X_nodes, X_edges, y in test_loader:
            y_pred = model(X_nodes, X_edges, G_in, G_out)
            y_true.append(y)
            y_pred_all.append(y_pred)

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred_all = torch.cat(y_pred_all, dim=0).numpy()

    # 计算微平均 ROC 曲线和 AUC，确保 y_true 中有正样本
    if np.sum(y_true) > 0:  # 确保有正样本，避免全为负样本的情况
        fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_pred_all.ravel())
        roc_auc_micro = roc_auc_score(y_true, y_pred_all, average='micro')

        # 计算宏平均 ROC 曲线和 AUC
        fpr_macro = np.linspace(0, 1, 100)
        all_tpr = []
        valid_classes = []
        for i in range(y_true.shape[1]):
            if np.any(y_true[:, i] == 1) and np.any(y_true[:, i] == 0):  # 类别 i 中有正负样本
                valid_classes.append(i)
                fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_all[:, i])
                all_tpr.append(np.interp(fpr_macro, fpr, tpr))

        if all_tpr:  # 如果有任何有效的类别用于计算宏平均
            mean_tpr_macro = np.mean(all_tpr, axis=0)
            roc_auc_macro = roc_auc_score(y_true[:, valid_classes], y_pred_all[:, valid_classes], average='macro')

            # 绘制微平均 ROC 曲线
            plt.figure()
            plt.plot(fpr_micro, tpr_micro, label=f'Micro-average ROC curve (AUC = {roc_auc_micro:.2f})', color='blue',
                     linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Micro-average ROC Curve')
            plt.legend(loc="lower right")
            plt.show()

            # 绘制宏平均 ROC 曲线
            plt.figure()
            plt.plot(fpr_macro, mean_tpr_macro, label=f'Macro-average ROC curve (AUC = {roc_auc_macro:.2f})',
                     color='green', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Macro-average ROC Curve')
            plt.legend(loc="lower right")
            plt.show()

            # 输出宏平均和微平均的 ROC-AUC
            print(f'Micro-average ROC-AUC score: {roc_auc_micro:.4f}')
            print(f'Macro-average ROC-AUC score: {roc_auc_macro:.4f}')
        else:
            print("No valid classes available for macro-average ROC AUC.")
    else:
        print("No positive samples available for micro-average ROC AUC.")

    return y_true, y_pred_all


def find_best_threshold(y_true, y_pred_all):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_thresholds = np.ones(y_true.shape[1]) * 0.5  # 初始化为0.5
    best_f1_scores = np.zeros(y_true.shape[1])

    for i in range(y_true.shape[1]):  # 针对每个节点单独调整阈值
        for threshold in thresholds:
            y_pred_labels = (y_pred_all[:, i] > threshold).astype(int)
            f1 = f1_score(y_true[:, i], y_pred_labels)
            if f1 > best_f1_scores[i]:
                best_f1_scores[i] = f1
                best_thresholds[i] = threshold

    return best_thresholds, best_f1_scores

if __name__ == '__main__':
    # 训练模型
    train(epochs, model, loss, optimizer, scheduler, train_dataloader, G_in, G_out)

    # 评估模型并获取最佳阈值
    y_true, y_pred_all = evaluate(model, test_dataloader, G_in, G_out)
    best_thresholds, best_f1_scores = find_best_threshold(y_true, y_pred_all)

    print(f'Best Thresholds: {best_thresholds}')
    print(f'Best F1 Scores: {best_f1_scores}')

    y_pred_labels = np.zeros_like(y_pred_all)
    for i in range(y_pred_all.shape[1]):
        y_pred_labels[:, i] = (y_pred_all[:, i] > best_thresholds[i]).astype(int)

    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels, average='micro')
    recall = recall_score(y_true, y_pred_labels, average='micro')
    f1_micro = f1_score(y_true, y_pred_labels, average='micro')
    f1_weighted = f1_score(y_true, y_pred_labels, average='weighted')
    hamming = hamming_loss(y_true, y_pred_labels)
    jaccard = jaccard_score(y_true, y_pred_labels, average='samples')

    print(f"Final Evaluation with Best Thresholds:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Micro Precision: {precision:.4f}")
    print(f"Micro Recall: {recall:.4f}")
    print(f"Micro F1 Score: {f1_micro:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Jaccard Index: {jaccard:.4f}")

    report = classification_report(y_true, y_pred_labels, zero_division=1, target_names=[f'Node {i+1}' for i in range(n_node)])
    print(f"Classification Report:\n{report}")
