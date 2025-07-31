# Hybrid Recommendation Experiment: GRPO/DPO Ranking + Bandit Exploration

# ---
# Cell 1: 场景设定与结构说明（Markdown）
"""
# 推荐系统中的混合排序与 Bandit/RL 探索实验

## 🎯 实验背景

本实验模拟了一个工业推荐系统的典型架构，结合主流的 pairwise ranking（如 GRPO/DPO 损失）和探索型 bandit 策略，用于兼顾大规模排序和冷启动/长期收益。

- 主体模型：GRPO/DPO 双塔结构，pairwise preference 排序训练
- 探索策略：为新用户/新物品引入 epsilon-greedy bandit 推荐
- Session 模拟：用户多轮交互，累计长期 reward，支持 RL-style 评估

---

## 🏗️ 模型与策略结构

1. **主召回排序模块**：用 pairwise loss 优化 embedding（双塔模型）
2. **Bandit 探索模块**：对冷启动 user/item 用 epsilon-greedy 策略推荐（探索 vs 利用）
3. **Session 多轮仿真**：每轮推荐后采集反馈，累计 reward，分析主 ranking 与探索 bandit 表现差异

---

## 📊 评估指标
- Pairwise 排序准确率
- 各用户组累计 reward
- bandit 策略分流下探索与 exploitation 比例
- session CTR/活跃等长期指标
"""

# ---
# Cell 2: 数据模拟与多轮 session 构造
import numpy as np
import pandas as pd
np.random.seed(42)

n_users, n_items = 200, 300
user_ids = [f'user_{i}' for i in range(n_users)]
item_ids = [f'item_{j}' for j in range(n_items)]

# 标记部分用户/物品为冷启动
cold_start_users = set(np.random.choice(user_ids, size=20, replace=False))
cold_start_items = set(np.random.choice(item_ids, size=30, replace=False))

# 生成用户行为日志（多 session，每 session 3~6次推荐）
sessions = []
for u in user_ids:
    n_sessions = np.random.randint(2, 6)
    for s in range(n_sessions):
        session = []
        for t in range(np.random.randint(3, 7)):
            i = np.random.choice(item_ids)
            click = np.random.randint(0, 4)  # 点击次数
            buy = np.random.choice([0, 1], p=[0.94, 0.06])
            stay = np.random.exponential(2)
            dislike = np.random.choice([0, 1], p=[0.97, 0.03])
            session.append({'user': u, 'item': i, 'click': click, 'buy': buy, 'stay': stay, 'dislike': dislike})
        sessions.append(session)

# 构造 pairwise 偏好三元组（正样本：多次点击/购买/长停留，负样本：点👎）
tuple_list = []
for session in sessions:
    u = session[0]['user']
    pos = [x['item'] for x in session if ((x['click'] >= 2) or (x['buy'] == 1) or (x['stay'] > 3)) and x['dislike']==0]
    neg = [x['item'] for x in session if x['dislike']==1]
    if len(pos) > 0 and len(neg) > 0:
        n = min(len(pos), len(neg))
        tuple_list += [(u, pos[i % len(pos)], neg[i % len(neg)]) for i in range(n)]

print(f"pairwise三元组样本数: {len(tuple_list)}")

# ---
# Cell 3: 特征模拟
embedding_dim = 32
user_feat = {u: np.random.normal(size=(embedding_dim,)).astype(np.float32) for u in user_ids}
item_feat = {i: np.random.normal(size=(embedding_dim,)).astype(np.float32) for i in item_ids}

# ---
# Cell 4: 双塔模型结构 + Ranking 训练（PyTorch）
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Tower(nn.Module):
    def __init__(self, in_dim, out_dim=32):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.fc(x))

class GRPOModel(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.user_tower = Tower(embed_dim, embed_dim)
        self.item_tower = Tower(embed_dim, embed_dim)
    def forward(self, u, i):
        u_emb = self.user_tower(u)
        i_emb = self.item_tower(i)
        return (u_emb * i_emb).sum(-1)

model = GRPOModel(embedding_dim).to(device)

# ---
# Cell 5: Pairwise Dataset & DataLoader
from torch.utils.data import Dataset, DataLoader

class PairwiseDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets
    def __len__(self):
        return len(self.triplets)
    def __getitem__(self, idx):
        u, ip, ineg = self.triplets[idx]
        return (
            torch.tensor(user_feat[u]),
            torch.tensor(item_feat[ip]),
            torch.tensor(item_feat[ineg])
        )

dataset = PairwiseDataset(tuple_list)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# ---
# Cell 6: GRPO/DPO 损失函数

def grpo_loss(pos_score, neg_score, beta=0.1):
    kl = beta * (pos_score.pow(2).mean() + neg_score.pow(2).mean())
    pref = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
    return pref + kl

def dpo_loss(pos_score, neg_score, beta=0.1):
    margin = (pos_score - neg_score) / beta
    return -torch.log(torch.sigmoid(margin)).mean()

# ---
# Cell 7: Ranking 训练主循环
from tqdm import tqdm
import matplotlib.pyplot as plt

epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_curve = []
acc_curve = []

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
        u, ip, ineg = [x.float().to(device) for x in batch]
        pos_score = model(u, ip)
        neg_score = model(u, ineg)
        #loss = grpo_loss(pos_score, neg_score, beta=0.1)
        loss = dpo_loss(pos_score, neg_score, beta=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * u.shape[0]
        correct += (pos_score > neg_score).sum().item()
        total += u.shape[0]
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    loss_curve.append(epoch_loss)
    acc_curve.append(epoch_acc)
    print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, pairwise acc={epoch_acc:.4f}")

# ---
# Cell 8: Bandit (epsilon-greedy) 探索策略实现与评估

# 简单模拟：每个 session 内
# 对主 ranking 用户用排序最高得分推荐，对 cold_start 用户 epsilon 概率随机探索
np.random.seed(43)
def recommend_bandit(user, user_emb, candidate_items, model, eps=0.2):
    if user in cold_start_users:
        if np.random.rand() < eps:
            # 探索（随机推荐）
            return np.random.choice(candidate_items)
        # 利用（用模型推荐）
    # 排序利用（直接用模型输出分数）
    user_vec = torch.tensor(user_emb[user]).float().unsqueeze(0).to(device)
    item_vecs = torch.stack([torch.tensor(item_feat[i]).float() for i in candidate_items]).to(device)
    scores = model.user_tower(user_vec) @ model.item_tower(item_vecs).T
    top_idx = torch.argmax(scores, dim=1).item()
    return candidate_items[top_idx]

# 统计探索分流、累计 reward
bandit_stats = {"explore":0, "exploit":0, "reward":[], "cold":[], "warm":[]}
session_reward_curve = []

for session in sessions[:1500]: # 只模拟部分session节省时间
    u = session[0]['user']
    candidate_items = [x['item'] for x in session]
    ground_truth = {(x['item'], x['buy'] or (x['click']>=2) or (x['stay']>3)) for x in session}
    recommended = recommend_bandit(u, user_feat, candidate_items, model, eps=0.2)
    # 简化 reward: 若推荐命中正样本 reward=1，否则0
    reward = 1 if (recommended, True) in ground_truth else 0
    if u in cold_start_users and np.random.rand() < 0.2:
        bandit_stats["explore"] += 1
    else:
        bandit_stats["exploit"] += 1
    bandit_stats["reward"].append(reward)
    (bandit_stats["cold"] if u in cold_start_users else bandit_stats["warm"]).append(reward)
    session_reward_curve.append(np.mean(bandit_stats["reward"]))

print(f"探索分流次数: {bandit_stats['explore']}, 利用次数: {bandit_stats['exploit']}")
print(f"总体平均 reward: {np.mean(bandit_stats['reward']):.3f}")
print(f"冷启动用户 reward: {np.mean(bandit_stats['cold']):.3f}, 主流用户 reward: {np.mean(bandit_stats['warm']):.3f}")

# ---
# Cell 9: Loss & Reward 曲线可视化
plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.plot(loss_curve, label='Ranking Loss')
plt.xlabel('Epoch'); plt.legend(); plt.title('Ranking Loss')
plt.subplot(1,3,2)
plt.plot(acc_curve, label='Pairwise Accuracy')
plt.xlabel('Epoch'); plt.legend(); plt.title('Pairwise Accuracy')
plt.subplot(1,3,3)
plt.plot(session_reward_curve, label='Bandit Mean Reward')
plt.xlabel('Session'); plt.legend(); plt.title('Bandit Mean Reward')
plt.tight_layout()
plt.show()

# ---
# Cell 10: 总结与建议（Markdown）
"""
## ✅ 实验总结
- 本实验实现了工业推荐系统中主流 ranking+bandit 探索混合架构，既保证大规模排序能力，也支持冷启动/探索。
- 结果展示了 pairwise ranking 学习曲线，bandit 分流、冷启动和主流用户 reward 对比等。

### 📈 后续可扩展：
- 支持更复杂的 RL 策略（如 Thompson Sampling, policy gradient）
- 用户行为真实序列建模，长期 session reward/用户留存
- 与真实工业日志/线上AB实验对接评估
"""
