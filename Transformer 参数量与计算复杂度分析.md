# Transformer 参数量与计算复杂度分析

## 1. 参数量估算

设：
- `d` = 隐藏维度 (hidden size)
- `L` = Transformer 堆叠层数
- `d_ff` = FFN 中间层维度，通常取 `d_ff ≈ 4d`

### (a) Self-Attention
- 需要权重矩阵：
  - Query (Q): `d × d`
  - Key (K): `d × d`
  - Value (V): `d × d`
  - Output projection (O): `d × d`
- 参数总数 ≈ `4d²`

### (b) Feed-Forward Network (FFN)
- 默认是两层：
  - 第一层: `d × d_ff`
  - 第二层: `d_ff × d`
- 参数总数 ≈ `2 × d × d_ff`
- 当 `d_ff ≈ 4d` 时，≈ `8d²`

### (c) 总体参数量
- 每层参数量 ≈ `attention (4d²) + FFN (2dd_ff)`
- 总参数量 ≈ `O(L × (d² + d × d_ff))`

👉 常见配置 `d_ff = 4d` 时：  
总参数量 ≈ `O(L × d²)`

---

## 2. 计算复杂度估算

设：
- `n` = 序列长度 (tokens 数量)

### (a) Attention 部分
- 计算注意力分数：`Q × Kᵀ`  
  - Q 大小：`n × d`  
  - K 大小：`n × d`  
  - 复杂度 ≈ `O(n² × d)`
- 乘以 Value：`(n × n) × (n × d)`  
  - 复杂度 ≈ `O(n² × d)`

### (b) FFN 部分
- 复杂度 ≈ `O(n × d × d_ff)`

### (c) 总体复杂度
- Self-Attention: `O(n² × d)`
- FFN: `O(n × d × d_ff)`
- 每层总复杂度：`O(n² × d + n × d × d_ff)`

👉 当序列长度 n 很大时，**注意力的 O(n² × d) 是主要瓶颈**。

---