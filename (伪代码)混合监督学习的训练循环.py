import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ===== Step 1. 定义模型 (ViT) =====
# 这里用 torchvision 自带的 ViT
vit = models.vit_b_16(pretrained=False)   # backbone
num_classes = 1000  # 假设 ImageNet
vit.heads = nn.Linear(vit.heads.head.in_features, num_classes)  # 分类头

# ===== Step 2. 定义损失函数 =====
ce_loss = nn.CrossEntropyLoss()

def mim_loss(student_patches, teacher_patches, mask):
    """对比 masked patch token 表征 (简化版 KL Loss)"""
    # 只在 mask 部分计算 loss
    mask = mask.unsqueeze(-1)  # [batch, num_patches, 1]
    return F.mse_loss(student_patches * mask, teacher_patches * mask)

# ===== Step 3. Teacher 模型 (动量更新) =====
teacher = models.vit_b_16(pretrained=False)
teacher.heads = nn.Identity()  # 去掉分类头
teacher.load_state_dict(vit.state_dict())
for p in teacher.parameters():
    p.requires_grad = False

def update_teacher(student, teacher, momentum=0.996):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data = momentum * pt.data + (1 - momentum) * ps.data

# ===== Step 4. 训练循环 =====
optimizer = torch.optim.Adam(vit.parameters(), lr=1e-4)

for epoch in range(epochs):
    for (x_labeled, y) , (x_unlabeled, mask) in zip(labeled_loader, unlabeled_loader):
        # -------- Supervised Loss --------
        logits = vit(x_labeled)  # [batch, num_classes]
        loss_sup = ce_loss(logits, y)

        # -------- Self-Supervised Loss (MIM) --------
        # student 表征
        student_features = vit._process_input(x_unlabeled)  # [batch, num_patches, dim]
        student_tokens = vit.encoder(student_features)      # patch tokens
        # teacher 表征 (detach)
        with torch.no_grad():
            teacher_features = teacher._process_input(x_unlabeled)
            teacher_tokens = teacher.encoder(teacher_features)
        loss_ssl = mim_loss(student_tokens, teacher_tokens, mask)

        # -------- 总损失 --------
        loss = loss_sup + 0.5 * loss_ssl  # λ=0.5

        # -------- 反向传播 --------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -------- 更新 Teacher --------
        update_teacher(vit, teacher, momentum=0.996)

    print(f"Epoch {epoch}: Supervised Loss={loss_sup.item():.4f}, SSL Loss={loss_ssl.item():.4f}")
