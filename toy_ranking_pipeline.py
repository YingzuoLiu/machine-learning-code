import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ======================
# 1. 合成数据
# ======================

np.random.seed(42)
torch.manual_seed(42)

N_USERS = 1000
N_ITEMS = 500
USER_DIM = 8
ITEM_DIM = 8

users = np.random.randn(N_USERS, USER_DIM)
items = np.random.randn(N_ITEMS, ITEM_DIM)

def generate_samples(n=5000):
    u_idx = np.random.randint(0, N_USERS, n)
    i_idx = np.random.randint(0, N_ITEMS, n)

    u = users[u_idx]
    it = items[i_idx]
    x = np.concatenate([u, it], axis=1)

    # relevance / ctr
    ctr_logit = (u * it).sum(axis=1)
    ctr = (ctr_logit + np.random.randn(n) * 0.5 > 0).astype(np.float32)

    # cvr
    cvr_logit = ctr_logit - 0.5
    cvr = (cvr_logit + np.random.randn(n) > 0).astype(np.float32) * ctr

    # dwell time（连续值）
    dwell = np.clip(ctr_logit + np.random.randn(n), 0, None)

    return x, ctr, cvr, dwell

X, y_ctr, y_cvr, y_dwell = generate_samples()

dataset = TensorDataset(
    torch.tensor(X, dtype=torch.float32),
    torch.tensor(y_ctr),
    torch.tensor(y_cvr),
    torch.tensor(y_dwell),
)

loader = DataLoader(dataset, batch_size=128, shuffle=True)

# ======================
# 2. 多任务 Ranking 模型
# ======================

class MultiTaskRanker(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.ctr_head = nn.Linear(32, 1)
        self.cvr_head = nn.Linear(32, 1)
        self.dwell_head = nn.Linear(32, 1)

    def forward(self, x):
        h = self.shared(x)
        return {
            "ctr": torch.sigmoid(self.ctr_head(h)),
            "cvr": torch.sigmoid(self.cvr_head(h)),
            "dwell": torch.relu(self.dwell_head(h)),
        }

model = MultiTaskRanker(USER_DIM + ITEM_DIM)
opt = optim.Adam(model.parameters(), lr=1e-3)

bce = nn.BCELoss()
mse = nn.MSELoss()

# ======================
# 3. 训练（多 loss）
# ======================

for epoch in range(5):
    total = 0
    for x, ctr, cvr, dwell in loader:
        out = model(x)
        loss = (
            1.0 * bce(out["ctr"].squeeze(), ctr)
            + 0.5 * bce(out["cvr"].squeeze(), cvr)
            + 0.1 * mse(out["dwell"].squeeze(), dwell)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {epoch}, loss={total:.2f}")

# ======================
# 4. 推理 + Ranking Pipeline
# ======================

def ranking_pipeline(user_id, topk=10):
    u = users[user_id]
    feats = np.concatenate([np.repeat(u[None], N_ITEMS, axis=0), items], axis=1)
    feats = torch.tensor(feats, dtype=torch.float32)

    with torch.no_grad():
        pred = model(feats)

    ctr = pred["ctr"].squeeze().numpy()
    cvr = pred["cvr"].squeeze().numpy()
    dwell = pred["dwell"].squeeze().numpy()

    # --- 粗排（pointwise / CTR）
    coarse_idx = np.argsort(-ctr)[:100]

    # --- 精排 base score
    base_score = (
        0.6 * ctr[coarse_idx]
        + 0.3 * cvr[coarse_idx]
        + 0.1 * dwell[coarse_idx]
    )

    # --- 模拟业务 penalty（比如补贴成本）
    cost = np.random.rand(len(coarse_idx)) * 0.2
    final_score = base_score - cost

    # --- 模拟新商家 quota
    is_new = np.random.rand(len(coarse_idx)) < 0.2
    final_rank = np.argsort(-final_score)

    # quota：TopK 至少 20% 新商家
    res = []
    new_quota = int(topk * 0.2)
    for idx in final_rank:
        if len(res) >= topk:
            break
        if is_new[idx] and sum(is_new[i] for i in res) < new_quota:
            res.append(idx)
        elif not is_new[idx]:
            res.append(idx)

    print("\nTopK results:")
    for r in res:
        print(
            f"item={coarse_idx[r]:3d} "
            f"ctr={ctr[coarse_idx[r]]:.2f} "
            f"cvr={cvr[coarse_idx[r]]:.2f} "
            f"dwell={dwell[coarse_idx[r]]:.2f} "
            f"new={is_new[r]}"
        )

ranking_pipeline(user_id=0)
