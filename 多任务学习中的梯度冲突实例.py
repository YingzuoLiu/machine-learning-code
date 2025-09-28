import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== 1. 定义多任务模型 ======
class MultiTaskAdModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_embedding = nn.Linear(100, 64)  # 100维 -> 64维共享表示
        self.ctr_head = nn.Linear(64, 1)  # CTR预测头
        self.cvr_head = nn.Linear(64, 1)  # CVR预测头
        
    def forward(self, x):
        shared_features = self.shared_embedding(x)
        ctr_logits = self.ctr_head(shared_features)
        cvr_logits = self.cvr_head(shared_features)
        return ctr_logits, cvr_logits


# ====== 2. 梯度计算函数 ======
def compute_task_gradients(model, batch):
    x, ctr_labels, cvr_labels = batch

    # ---- CTR梯度 ----
    model.zero_grad()
    ctr_pred, _ = model(x)
    ctr_loss = F.binary_cross_entropy_with_logits(ctr_pred, ctr_labels)
    ctr_loss.backward(retain_graph=True)

    ctr_gradients = []
    for param in model.shared_embedding.parameters():  # 只收集共享层
        if param.grad is not None:
            ctr_gradients.append(param.grad.flatten())
    ctr_grad_vector = torch.cat(ctr_gradients).detach()

    # ---- CVR梯度 ----
    model.zero_grad()
    _, cvr_pred = model(x)
    cvr_loss = F.binary_cross_entropy_with_logits(cvr_pred, cvr_labels)
    cvr_loss.backward()

    cvr_gradients = []
    for param in model.shared_embedding.parameters():  # 只收集共享层
        if param.grad is not None:
            cvr_gradients.append(param.grad.flatten())
    cvr_grad_vector = torch.cat(cvr_gradients).detach()

    return ctr_grad_vector, cvr_grad_vector


# ====== 3. 测试代码 ======
if __name__ == "__main__":
    torch.manual_seed(42)

    # 模拟batch (batch_size=8, feature_dim=100)
    x = torch.randn(8, 100)
    ctr_labels = torch.randint(0, 2, (8, 1)).float()
    cvr_labels = torch.randint(0, 2, (8, 1)).float()

    model = MultiTaskAdModel()

    ctr_grad_vector, cvr_grad_vector = compute_task_gradients(model, (x, ctr_labels, cvr_labels))

    # 防止梯度全零导致 NaN
    if torch.norm(ctr_grad_vector) > 0 and torch.norm(cvr_grad_vector) > 0:
        cos_similarity = F.cosine_similarity(ctr_grad_vector, cvr_grad_vector, dim=0)
    else:
        cos_similarity = torch.tensor(0.0)

    print(f"梯度余弦相似度: {cos_similarity:.3f}")
    if cos_similarity < 0:
        print("⚠️ 检测到梯度冲突，两个任务的方向大致相反！")
    else:
        print("✅ 梯度方向基本一致或无冲突。")
