# 可逆流（Normalizing Flow）推导笔记

本文以 **图像建模** 为例，逐步展开 **可逆流生成模型**（Normalizing Flow）的推导过程，并附上二维数据与数值示例，同时说明如何推广到高维图像场景（如 CIFAR-10 的 32×32×3 图像）。

---

## 1. 基本思想

我们希望学习一个 **数据分布** \(p_{\text{data}}(x)\)，其中：
- \(x \in \mathbb{R}^{d}\)：数据样本，例如一张图像展平后的像素向量。
- 目标：构建一个生成模型，既能 **评估概率密度**，也能 **生成新样本**。

**核心想法**：
- 构造一个 **可逆映射** \(f_\theta\)，把数据 \(x\) 映射到一个简单分布（如高斯分布）。
- 通过可逆性，可以在 **数据空间**和**潜在空间**之间来回转换。

---

## 2. 可逆映射函数

我们定义一个 **双射函数**：

\[
z = f_\theta(x), \quad x = f_\theta^{-1}(z)
\]

性质：
1. 可逆性：每个 \(x\) 对应唯一的 \(z\)，反之亦然。
2. 光滑性：保证 Jacobian 可计算。

在图像例子中：
- \(x\)：一个 32×32×3 的彩色图像展平成向量，维度 \(d = 3072\)。
- \(z\)：同维度的潜在变量（通常假设来自标准正态分布）。

---

## 3. 概率密度变换（Change of Variables）

给定 \(z \sim p_Z(z) = \mathcal{N}(0, I)\)，数据分布通过变换公式得到：

\[
p_X(x) = p_Z(f_\theta(x)) \cdot \left| \det \frac{\partial f_\theta(x)}{\partial x} \right|
\]

其中：
- 第一项：潜在变量的密度。
- 第二项：Jacobian 行列式，修正体积变化。

取对数：

\[
\log p_X(x) = \log p_Z(z) + \log \left| \det J_{f_\theta}(x) \right|
\]

---

## 4. 训练目标

训练时使用 **最大似然估计（MLE）**：

\[
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \log p_X(x_i)
\]

其中 \(x_i\) 是训练图像样本。

---

## 5. 在图像建模中的应用

1. **训练阶段**：
   - 输入真实图像 \(x\)。
   - 通过 \(z = f_\theta(x)\) 得到潜在表示。
   - 计算 \(\log p_X(x)\)，更新参数 \(\theta\)。

2. **生成阶段**：
   - 从高斯分布采样 \(z \sim \mathcal{N}(0,I)\)。
   - 通过 \(x = f_\theta^{-1}(z)\) 得到生成图像。

---

## 6. 推导展开：从数据到潜在空间

以图像 \(x \in \mathbb{R}^{d}\) 为例：

1. **变换到潜在空间**：

\[
z = f_\theta(x)
\]

2. **潜在变量分布**（标准正态）：

\[
p_Z(z) = \frac{1}{(2\pi)^{d/2}} \exp\left(-\tfrac{1}{2}\|z\|^2 \right)
\]

3. **数据分布的显式公式**：

\[
p_X(x) = \frac{1}{(2\pi)^{d/2}} \exp\left(-\tfrac{1}{2}\|f_\theta(x)\|^2 \right) \cdot \left| \det J_{f_\theta}(x) \right|
\]

4. **对数似然**：

\[
\log p_X(x) = -\tfrac{1}{2}\|f_\theta(x)\|^2 - \tfrac{d}{2}\log(2\pi) + \log \left| \det J_{f_\theta}(x) \right|
\]

---

## 7. 关键问题

- **如何设计 \(f_\theta\)，使得 Jacobian 的行列式高效可计算？**
  - NICE：加性耦合层。
  - RealNVP：仿射耦合层。
  - Glow：1×1 可逆卷积。

- **优点**：
  - 可以精确计算似然（不同于 GAN）。
  - 可以双向映射（不同于 VAE）。

- **缺点**：
  - 设计受限，必须保证可逆且行列式易算。

---

## 8. 总结

可逆流模型的核心推导步骤：
1. 定义可逆映射 \(f_\theta\)。
2. 应用变量替换公式，得到 \(p_X(x)\)。
3. 使用 MLE 最大化对数似然进行训练。
4. 在生成阶段，采样 \(z\) 并反变换得到 \(x\)。

---

## 9. 示例：二维可逆流（公式推导）

### 9.1 数据与潜在变量
- 数据点：\(x = (x_1, x_2) \in \mathbb{R}^2\)。
- 潜在变量：\(z = (z_1, z_2) \sim \mathcal{N}(0, I)\)。

### 9.2 仿射耦合变换

\[
\begin{cases}
 z_1 = x_1 \\
 z_2 = x_2 \cdot \exp(s(x_1)) + t(x_1)
\end{cases}
\]

### 9.3 Jacobian

\[
J = \frac{\partial (z_1, z_2)}{\partial (x_1, x_2)} =
\begin{bmatrix}
1 & 0 \\
\tfrac{\partial z_2}{\partial x_1} & \exp(s(x_1))
\end{bmatrix},
\quad \det J = \exp(s(x_1))
\]

### 9.4 对数似然

\[
\log p_X(x) = -\tfrac{1}{2}(z_1^2 + z_2^2) - \log(2\pi) + s(x_1)
\]

### 9.5 生成过程

\[
\begin{cases}
 x_1 = z_1 \\
 x_2 = (z_2 - t(x_1)) \cdot \exp(-s(x_1))
\end{cases}
\]

---

## 10. 示例：二维数值计算

### 10.1 线性仿射变换

设定：
- 矩阵 \(A = \begin{bmatrix}2 & 0.5 \\ 0.1 & 1.5\end{bmatrix}\)，偏置 \(b = [0.3, -0.2]\)。
- 输入 \(x = [0.5, -1.0]^T\)。

计算：
- \(z = A x + b = [0.8, -1.65]^T\)。
- \(\det A = 2.95, \quad \log |\det A| \approx 1.0818\)。
- \(\|z\|^2 = 3.3625\)。
- \(\log p_Z(z) \approx -3.5191\)。
- \(\log p_X(x) = -2.4373\)。

### 10.2 仿射耦合层（RealNVP 原型）

设定：
- \(s(x_1) = 0.5 x_1, \; t(x_1) = 0.2 x_1\)。
- 输入 \(x = (0.5, -1.0)\)。

计算：
- \(y_1 = 0.5\)， \(y_2 = -1.0 e^{0.25} + 0.1 \approx -1.1840\)。
- \(\det J = e^{0.25} \approx 1.2840, \quad \log |\det J| = 0.25\)。
- \(\|y\|^2 \approx 1.6519\)。
- \(\log p_Z(y) \approx -2.6638\)。
- \(\log p_X(x) \approx -2.4138\)。

---

## 11. 从二维到高维图像（d=3072）的推广要点

二维 toy 例子和高维图像的推导公式 **完全一致**，但在实现上需要额外处理：

### 11.1 数据预处理
- **离散像素 → 连续化**：加入均匀噪声 \(u \sim \mathrm{Uniform}[0,1)\)，得到
  \[
  x' = (\text{pixel} + u)/256
  \]
- **Logit 变换**：为了映射到 \(\mathbb{R}^d\)，使用
  \[
  y = \mathrm{logit}(\alpha + (1-2\alpha)x')
  \]
  并加上对应的 \(\log |\det|\) 修正项。

### 11.2 高维 Jacobian 的高效计算
- 不能直接算 \(3072\times 3072\) 的行列式。
- **耦合层**：Jacobian 下三角，\(\log\det J = \sum s(\cdot)\)。
- **Glow 的 1×1 可逆卷积**：
  \[
  \log \det J = H \cdot W \cdot \log |\det W|
  \]
  其中 \(W\) 是通道变换矩阵。

### 11.3 多尺度结构
- 在若干层之后下采样或分离部分变量，减小后续计算负担。
- 每个阶段的 \(\log\det J\) 累加得到全局对数似然。

### 11.4 示例代码片段（PyTorch 耦合层）
```python
import torch
import torch.nn as nn

class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(dim//2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        st = self.nn(x1)
        s, t = st.chunk(2, dim=1)
        y2 = x2 * torch.exp(s) + t
        logdet = s.sum(dim=1)
        return torch.cat([x1, y2], dim=1), logdet
```

---


