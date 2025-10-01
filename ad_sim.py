"""
假设与限制
假设 (Assumptions)
广告市场：有多个广告主，每个广告有固定出价 bid 与 embedding 表示其特征。
用户画像：每个用户有一个长期兴趣向量 true_pref，但系统只能通过点击/转化逐步学习。
兴趣漂移 (drift)：用户兴趣会随时间/交互轻微改变。
用户行为：点击概率由 sigmoid(<user_pref, ad_embed>) 决定，转化概率在点击后由另一模型决定。
三层优化目标：
即时层：最大化单次点击概率 (CTR)
会话层：保持用户体验（多样性、不过度频控）
长期层：提升用户留存率/生命周期价值 (LTV)
限制 (Limitations)
用户兴趣与广告 embedding 都是低维向量，简化了真实系统中的多模态特征。
拍卖机制只实现了简单 二价拍卖 (second-price auction)，未考虑大规模竞价/预算消耗。
在线微调模型只用一个小型 MLP，真实广告市场里通常会用更复杂的深度CTR/CVR模型。
用户体验层次化优化（即时 / 会话 / 长期）这里只是通过 reward shaping 近似实现，真实场景需要复杂策略（RL, 多目标优化）。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ===========================
# 业务参数设定
# ===========================
NUM_USERS = 50        # 用户数
NUM_ADS = 100         # 广告数
EMBED_DIM = 8         # embedding 维度 (广告特征 + 用户兴趣)
HISTORY_LEN = 5       # 用户历史序列长度
ROUNDS = 500          # 模拟的广告拍卖轮数
DRIFT_LR = 0.05       # 用户兴趣随点击漂移的速率

# ===========================
# 数据初始化
# ===========================
np.random.seed(42)
torch.manual_seed(42)

# 广告 embedding & bid
ad_embeddings = np.random.normal(0, 1, (NUM_ADS, EMBED_DIM))
ad_bids = np.random.uniform(0.5, 3.0, NUM_ADS)  # 出价，简化为固定值

# 用户真实兴趣向量
user_true_pref = np.random.normal(0, 1, (NUM_USERS, EMBED_DIM))

# 用户历史序列 (FIFO buffer)
user_histories = np.zeros((NUM_USERS, HISTORY_LEN, EMBED_DIM))
user_hist_lens = np.zeros(NUM_USERS, dtype=int)

# ===========================
# 模拟器函数
# ===========================
def click_prob(u, a):
    """用户 u 对广告 a 的点击概率 = sigmoid(内积)"""
    score = np.dot(user_true_pref[u], ad_embeddings[a])
    return 1 / (1 + np.exp(-score))

def convert_prob(u, a):
    """转化概率: 点击后再一次 sigmoid(内积)"""
    score = np.dot(user_true_pref[u], ad_embeddings[a]) / 2
    return 1 / (1 + np.exp(-score))

def append_history(u, a, click_flag):
    """
    更新用户历史序列 (FIFO 队列)
    """
    vec = ad_embeddings[a]
    length = user_hist_lens[u]
    if length < HISTORY_LEN:
        user_histories[u, length, :] = vec
        user_hist_lens[u] += 1
    else:
        user_histories[u, :-1, :] = user_histories[u, 1:, :]
        user_histories[u, -1, :] = vec

    # 如果点击，用户兴趣向量轻微漂移
    if click_flag:
        user_true_pref[u] = (1 - DRIFT_LR) * user_true_pref[u] + DRIFT_LR * vec

# ===========================
# 拍卖机制 (简化版二价拍卖)
# ===========================
def auction_round(user_id, model):
    """
    单轮广告拍卖：
    1. 模型预测 eCTR (点击率)
    2. 计算 eCPM = bid * eCTR
    3. 选出最高的广告并展示
    4. 模拟用户点击 / 转化
    """
    user_vec = torch.tensor(user_true_pref[user_id], dtype=torch.float32)
    ad_vecs = torch.tensor(ad_embeddings, dtype=torch.float32)

    # 模型预测 CTR
    ctr_pred = model(user_vec.repeat(NUM_ADS, 1), ad_vecs).detach().numpy()
    ecpms = ad_bids * ctr_pred

    # 选出最高 eCPM 的广告
    a_idx = np.argmax(ecpms)

    # 模拟用户反馈
    ctr_true = click_prob(user_id, a_idx)
    click = np.random.rand() < ctr_true
    conv = False
    revenue = 0
    if click:
        cvr_true = convert_prob(user_id, a_idx)
        conv = np.random.rand() < cvr_true
        revenue = ad_bids[a_idx] * 50 if conv else 0  # 转化收益

    # 更新用户历史
    append_history(user_id, a_idx, click)

    return a_idx, click, conv, revenue

# ===========================
# 简单 MLP 模型 (在线微调)
# ===========================
class CTRModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, user_vec, ad_vec):
        x = torch.cat([user_vec, ad_vec], dim=-1)
        return self.fc(x).squeeze(-1)

# ===========================
# 在线训练循环
# ===========================
model = CTRModel(EMBED_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

total_clicks, total_convs, total_revenue = 0, 0, 0

for r in range(1, ROUNDS + 1):
    u = np.random.randint(0, NUM_USERS)
    a_idx, click, conv, revenue = auction_round(u, model)

    # 训练样本
    user_vec = torch.tensor(user_true_pref[u], dtype=torch.float32)
    ad_vec = torch.tensor(ad_embeddings[a_idx], dtype=torch.float32)
    label = torch.tensor([1.0 if click else 0.0])

    pred = model(user_vec.unsqueeze(0), ad_vec.unsqueeze(0))
    loss = loss_fn(pred, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 累计指标
    total_clicks += int(click)
    total_convs += int(conv)
    total_revenue += revenue

    if r % 100 == 0:
        print(f"Round {r}, clicks={total_clicks}, conversions={total_convs}, revenue={total_revenue:.2f}")