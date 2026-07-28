# ============================================================
# 线性回归 (MLP) 时序预测 —— 高度模仿 sequence.ipynb
# 训练集: df_train (已清洗, 列含 pm_ave, DatetimeIndex)
# 测试集: df_test  (已清洗, 列含 pm_ave)
# 不创建任何新数据集，仅从 df_train / df_test 提取张量
# ============================================================

import torch
from torch import nn
from d2l import torch as d2l

# ============================================================
# 1. 准备时间序列 (直接从已有 df_train / df_test 提取 pm_ave)
# ============================================================
x_train = torch.tensor(df_train['pm_ave'].values, dtype=torch.float32)
x_test  = torch.tensor(df_test['pm_ave'].dropna().values, dtype=torch.float32)

print(f'训练序列长度: {len(x_train)}  |  测试序列长度: {len(x_test)}')

# ============================================================
# 2. 标准化
#    pm_ave 量级 (0~700) 远大于 sequence.ipynb 合成数据 (~±1)
#    不标准化会导致梯度爆炸，训练不稳定
# ============================================================
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std
x_test  = (x_test  - mean) / std   # 用训练集统计量标准化测试集

# ============================================================
# 3. 构建自回归特征 (用过去 tau 步预测下一步)
#    与 sequence.ipynb 完全相同的手法
# ============================================================
tau = 4  # 回看窗口: 用过去 4 个小时预测下一个小时

# --- 训练特征 ---
# features_train[i] = [x_i, x_{i+1}, x_{i+2}, x_{i+3}],  label = x_{i+4}
n_train = len(x_train)
features_train = torch.zeros((n_train - tau, tau))
for i in range(tau):
    features_train[:, i] = x_train[i: n_train - tau + i]
labels_train = x_train[tau:].reshape((-1, 1))

# --- 测试特征 (单步预测: 用真实观测值作为输入) ---
# 用训练集末尾 tau 步作为"桥接"，拼接训练末尾 + 测试全部
bridge = torch.cat([x_train[-tau:], x_test])
n_test = len(x_test)
features_test = torch.zeros((n_test, tau))
for i in range(tau):
    features_test[:, i] = bridge[i: i + n_test]
labels_test = x_test.reshape((-1, 1))

# ============================================================
# 4. 数据迭代器
# ============================================================
batch_size = 16
train_iter = d2l.load_array((features_train, labels_train), batch_size, is_train=True)

# ============================================================
# 5. 定义网络 + 损失 (与 sequence.ipynb 完全一致)
# ============================================================
# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(tau, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失。注意：MSELoss 计算平方误差时不带系数 1/2
loss = nn.MSELoss(reduction='none')

# ============================================================
# 6. 训练 (与 sequence.ipynb 完全一致)
# ============================================================
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

# ============================================================
# 7. 测试集单步预测 + MSE
# ============================================================
# 单步预测: 每个测试点的输入是前 tau 步的真实观测值
onestep_preds = net(features_test).detach()

# 反标准化回原始量纲 (μg/m³)
preds_orig  = onestep_preds * std + mean
labels_orig = labels_test   * std + mean

# 逐条计算均方误差
test_mse = ((preds_orig - labels_orig) ** 2).mean()
print(f'\n测试集 MSE = {test_mse:.2f}')

# ============================================================
# 8. 绘图 (模仿 sequence.ipynb 的单步预测图)
# ============================================================
test_time = torch.arange(1, len(x_test) + 1, dtype=torch.float32)
d2l.plot([test_time, test_time],
         [labels_orig.numpy(), preds_orig.numpy()],
         'test timestep', 'PM2.5 (μg/m³)',
         legend=['actual', '1-step preds'], figsize=(10, 4))
