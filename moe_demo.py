import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class MoEConfig:
    d_model: int = 64
    d_hidden: int = 128
    num_experts: int = 4
    k: int = 2
    bias_lr: float = 1e-3  # 外环“旋钮”更新速度 ρ

class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )
    def forward(self, x):  # (B,T,D) -> (B,T,D)
        return self.net(x)

class SimpleMoE(nn.Module):
    """
    Aux-loss-free 负载均衡：
      - 只在 Top-K 选择用 (raw + bias)
      - gate 归一化只用 raw（不含 bias）
      - 每 step 末：b_e <- b_e - ρ * (n_e - NK/E)
    """
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        E = cfg.num_experts
        self.router = nn.Linear(cfg.d_model, E)
        self.experts = nn.ModuleList([ExpertFFN(cfg.d_model, cfg.d_hidden) for _ in range(E)])
        # 不进计算图的偏置（便于保存/加载），persistent=True 会存进 state_dict
        self.register_buffer("expert_bias", torch.zeros(E), persistent=True)

    @torch.no_grad()
    def update_bias(self, counts: torch.Tensor):
        """counts: (E,) 本 step 每个专家被选 token 数（含 K 路由汇总）"""
        target = counts.sum().float() / self.cfg.num_experts  # NK/E
        self.expert_bias -= self.cfg.bias_lr * (counts.float() - target)

    def forward(self, x):
        """
        x: (B,T,D)
        return: y (B,T,D), stats {'counts': Tensor(E,)}
        """
        B, T, D = x.shape
        E, K = self.cfg.num_experts, self.cfg.k

        raw = self.router(x)                             # (B,T,E)
        biased = raw + self.expert_bias.view(1,1,E)      # 仅用于 Top-K 选择
        _, topk_idx = torch.topk(biased, k=K, dim=-1)    # (B,T,K)

        # gate 用原始 raw（不含 bias），在选中K里 softmax
        raw_sel = torch.gather(raw, dim=-1, index=topk_idx)  # (B,T,K)
        gates = F.softmax(raw_sel, dim=-1)                   # (B,T,K)

        # 展平以便按专家聚合
        N = B * T
        flat_x = x.reshape(N, D)
        flat_topk = topk_idx.reshape(N, K)
        flat_gates = gates.reshape(N, K)

        # 统计被选次数（用于外环偏置更新）
        with torch.no_grad():
            counts = torch.bincount(flat_topk.view(-1), minlength=E)  # (E,)

        # 聚合各专家输出
        out_accum = torch.zeros(N, D, device=x.device)
        for e in range(E):
            hit = (flat_topk == e)
            if not hit.any():
                continue
            # 一个 token 可能在多个槽位选到同一专家 → 合并 gate
            gate_e = (flat_gates * hit.float()).sum(dim=-1)  # (N,)
            idx = gate_e.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            y_e = self.experts[e](flat_x.index_select(0, idx))       # (M,D)
            out_accum.index_add_(0, idx, y_e * gate_e.index_select(0, idx).unsqueeze(-1))

        y = out_accum.view(B, T, D)
        return y, {"counts": counts}
