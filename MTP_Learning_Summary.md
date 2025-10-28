# 🧠 MTP（Multi-Token Prediction）学习总结

> 适用背景：DeepSeek-V3、等语言模型多步预测机制  
> 关键词：多 Token 预测、密集监督信号、加速收敛、训练阶段增强

---

## 1️⃣ MTP 是什么？

**MTP（Multi-Token Prediction）** 是一种在训练阶段增强语言模型学习信号的机制。  
它让每个隐藏状态 `h_i` 不仅预测下一个词，还同时预测未来多个词。

传统语言模型只学：
\[
P(t_{i+1} \mid t_{\le i})
\]

而 MTP 学：
\[
P(t_{i+1}, t_{i+2}, ..., t_{i+K} \mid t_{\le i})
\]

这样每个 token 位置能获得来自多个未来 token 的梯度信号，  
提高训练效率、稳定性与长期依赖建模能力。

---

## 2️⃣ 为什么需要 MTP？

| 问题 | 传统 LM 的限制 | MTP 的改进 |
|------|----------------|-------------|
| 监督稀疏 | 每个隐藏状态只预测 1 个词 | 每个隐藏状态预测多个词 |
| 梯度单一 | 梯度只从最近目标反传 | 多个未来目标同时回传梯度 |
| 长期依赖弱 | 只能学习局部上下文 | 要预测多步 → 强制建模长期结构 |
| 收敛慢 | 每步更新信号少 | 每步多监督 → 更快收敛 |

---

## 3️⃣ 数学直觉

传统 LM 目标：
\[
\mathcal{L}_{NTP} = \sum_i \text{CE}(P(t_{i+1}), t_{i+1})
\]

MTP 目标：
\[
\mathcal{L}_{MTP} = 
\sum_i \sum_{k=1}^{K} 
\lambda_k \cdot \text{CE}\big(P(t_{i+k}\mid t_{\le i}), t_{i+k}\big)
\]

即每个隐藏状态 \(h_i\) 拥有多步预测头：
\[
h_i \rightarrow
\begin{cases}
\text{Head}_1: P(t_{i+1}) \\
\text{Head}_2: P(t_{i+2}) \\
\vdots \\
\text{Head}_K: P(t_{i+K})
\end{cases}
\]

最终 loss 对所有预测头做**加权平均**（或等权平均）：
\[
L = \frac{1}{K}\sum_{k=1}^{K} \lambda_k L_k
\]

---

## 4️⃣ 模型结构变化

### ✅ 不改变主干结构（Transformer Backbone）

- 注意力层、多层前馈层（FFN）完全不变。
- 改动仅在**输出层（head）**：
  - 从 1 个 `Linear + Softmax` → 多个并行的输出头。

| 模块 | 传统 LM | MTP |
|------|----------|------|
| Transformer 主体 | 不变 | 不变 |
| 输出层 | 1 个 Linear + Softmax | K 个 Linear + Softmax |
| Loss | 单一交叉熵 | 多步交叉熵加权平均 |
| 推理逻辑 | 逐词生成 | 仍然逐词生成 |

---

## 5️⃣ 训练阶段 vs 推理阶段

| 阶段 | 是否使用多头 | 是否平均 | 说明 |
|------|---------------|-----------|------|
| **训练阶段** | ✅ 使用多个预测头 | ✅ 对 loss 平均 | 增强监督信号 |
| **推理阶段** | ❌ 只用第一个预测头 | ❌ 不平均输出 | 正常自回归生成 |

**直觉类比**：  
训练时多做“练习题”（预测更多词），提高能力；  
推理时只做“考试题”（预测下一个词）。

---

## 6️⃣ PyTorch 实现示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MTPModel(nn.Module):
    def __init__(self, transformer, vocab_size, K=3):
        super().__init__()
        self.backbone = transformer       # 原始 Transformer，不变
        self.heads = nn.ModuleList([
            nn.Linear(transformer.hidden_size, vocab_size)
            for _ in range(K)
        ])
        self.K = K

    def forward(self, input_ids, targets):
        h = self.backbone(input_ids)      # 输出隐藏状态 [B, T, H]
        losses = []
        for k in range(1, self.K + 1):
            # 第 k 个预测头预测 t_{i+k}
            logits = self.heads[k - 1](h[:, :-k, :])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets[:, k:].reshape(-1)
            )
            losses.append(loss)
        # 平均多个预测头的 loss
        return sum(losses) / self.K
```

🧠 注意：
- 模型只在训练时使用多个输出头；
- 推理阶段只保留第一个 head；
- `loss` 平均，而非对 softmax 概率平均。

---

## 7️⃣ 结构示意对比

### 🔹 传统语言模型

```
t₁, t₂, t₃ → Transformer → h₁, h₂, h₃
                  ↓
             Linear + Softmax
                  ↓
           预测下一个词 t₄
```

### 🔹 MTP 模型

```
t₁, t₂, t₃ → Transformer → h₁, h₂, h₃
                  ↓
         ┌────────┼────────┐
         ↓        ↓        ↓
   Head₁(W₁)  Head₂(W₂)  Head₃(W₃)
     ↓           ↓           ↓
  t₍ᵢ₊₁₎       t₍ᵢ₊₂₎       t₍ᵢ₊₃₎
```

训练时计算：
\[
L = (L_1 + L_2 + L_3)/3
\]

推理时仅使用：
\[
\text{Head}_1
\]

---

## 8️⃣ 总结与启发

| 维度 | 内容 |
|------|-------|
| **本质** | 多头未来预测 → 增强训练监督信号 |
| **结构变化** | 仅在输出层，多头线性映射 |
| **数学目标** | 高阶条件概率建模 |
| **优化效果** | 更快收敛、更强长期依赖理解 |
| **推理影响** | 无变化（仍逐词生成） |
| **借鉴启发** | 在推荐、序列、强化学习中可用于“多步预测”或“多任务监督” |

---

### 🧩 一句话总结

> **MTP 不改变 Transformer 结构，只改变训练目标。**  
> 它让每个隐藏状态预测多个未来 token，  
> 在训练时获得更丰富的梯度反馈，  
> 提升模型学习效率与上下文建模能力。

---
