import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ============================================================================
# 1. KL Annealing 实现
# ============================================================================

class KLAnnealingScheduler:
    """KL权重退火调度器"""
    def __init__(self, anneal_type='linear', max_steps=10000, min_beta=0.0, max_beta=1.0):
        self.anneal_type = anneal_type
        self.max_steps = max_steps
        self.min_beta = min_beta
        self.max_beta = max_beta
    
    def get_beta(self, step):
        if step >= self.max_steps:
            return self.max_beta
        
        progress = step / self.max_steps
        
        if self.anneal_type == 'linear':
            beta = self.min_beta + (self.max_beta - self.min_beta) * progress
        
        elif self.anneal_type == 'cosine':
            beta = self.min_beta + (self.max_beta - self.min_beta) * 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        
        elif self.anneal_type == 'sigmoid':
            # sigmoid annealing: 慢启动，快结束
            s = 1.0 / (1.0 + math.exp(-12 * (progress - 0.5)))
            beta = self.min_beta + (self.max_beta - self.min_beta) * s
        
        elif self.anneal_type == 'cyclical':
            # 循环退火，适合长训练
            cycle = 4  # 每个周期的比例
            progress_in_cycle = (progress * cycle) % 1.0
            beta = self.min_beta + (self.max_beta - self.min_beta) * progress_in_cycle
        
        return beta

class KLAnnealingVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        self.kl_scheduler = KLAnnealingScheduler('sigmoid', max_steps=5000)
        self.current_step = 0
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # KL散度
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # 重建损失
        recon_loss = F.binary_cross_entropy(recon, x, reduction='none').sum(dim=1)
        
        # 动态β权重
        beta = self.kl_scheduler.get_beta(self.current_step)
        
        # 总损失
        loss = recon_loss + beta * kl
        
        return {
            'loss': loss.mean(),
            'recon_loss': recon_loss.mean(),
            'kl_loss': kl.mean(),
            'beta': beta,
            'recon': recon
        }
    
    def step(self):
        """训练步数更新"""
        self.current_step += 1

# ============================================================================
# 2. Free Bits 实现
# ============================================================================

class FreeBitsVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, free_bits=2.0):
        super().__init__()
        self.free_bits = free_bits  # 每个维度的免费bits
        
        # 网络结构同上
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def free_bits_kl(self, mu, logvar):
        """Free bits KL散度计算"""
        # 计算每个维度的KL散度
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
        
        # 应用free bits：max(free_bits, kl_per_dim)
        kl_free = torch.clamp(kl_per_dim, min=self.free_bits)
        
        return kl_free.sum(dim=1)  # 对latent维度求和
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # Free bits KL
        kl = self.free_bits_kl(mu, logvar)
        
        # 重建损失
        recon_loss = F.binary_cross_entropy(recon, x, reduction='none').sum(dim=1)
        
        # 总损失
        loss = recon_loss + kl
        
        # 计算实际使用的bits数
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
        active_dims = (kl_per_dim > self.free_bits).sum(dim=1).float()
        
        return {
            'loss': loss.mean(),
            'recon_loss': recon_loss.mean(),
            'kl_loss': kl.mean(),
            'active_dims': active_dims.mean(),
            'recon': recon
        }

# ============================================================================
# 3. Hierarchical VAE 实现
# ============================================================================

class HierarchicalVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dims=[32, 16, 8]):
        super().__init__()
        self.latent_dims = latent_dims
        self.n_levels = len(latent_dims)
        
        # 编码器：bottom-up
        self.encoders = nn.ModuleList()
        prev_dim = input_dim
        for i, latent_dim in enumerate(latent_dims):
            encoder = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.encoders.append(encoder)
            prev_dim = hidden_dim
        
        # 编码器输出层
        self.mu_layers = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in latent_dims
        ])
        self.logvar_layers = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in latent_dims
        ])
        
        # 先验网络：top-down
        self.prior_nets = nn.ModuleList()
        for i in range(self.n_levels - 1):
            # 每层的先验依赖于上层
            prior_net = nn.Sequential(
                nn.Linear(latent_dims[i+1], hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, latent_dims[i] * 2)  # mu + logvar
            )
            self.prior_nets.append(prior_net)
        
        # 解码器：top-down
        self.decoder = nn.Sequential(
            nn.Linear(sum(latent_dims), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Bottom-up encoding"""
        mus, logvars = [], []
        h = x
        
        for i in range(self.n_levels):
            h = self.encoders[i](h)
            mu = self.mu_layers[i](h)
            logvar = self.logvar_layers[i](h)
            mus.append(mu)
            logvars.append(logvar)
        
        return mus, logvars
    
    def get_prior_params(self, zs):
        """获取条件先验参数"""
        prior_mus, prior_logvars = [], []
        
        # 最顶层使用标准正态先验
        prior_mus.append(torch.zeros_like(zs[-1]))
        prior_logvars.append(torch.zeros_like(zs[-1]))
        
        # 其他层使用条件先验
        for i in range(self.n_levels - 2, -1, -1):
            prior_params = self.prior_nets[i](zs[i+1])
            mu_dim = self.latent_dims[i]
            prior_mu = prior_params[:, :mu_dim]
            prior_logvar = prior_params[:, mu_dim:]
            prior_mus.append(prior_mu)
            prior_logvars.append(prior_logvar)
        
        # 翻转顺序以匹配编码器输出
        return list(reversed(prior_mus)), list(reversed(prior_logvars))
    
    def reparameterize(self, mus, logvars):
        zs = []
        for mu, logvar in zip(mus, logvars):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            zs.append(z)
        return zs
    
    def decode(self, zs):
        # 连接所有层的潜变量
        z_concat = torch.cat(zs, dim=1)
        return self.decoder(z_concat)
    
    def forward(self, x):
        # 编码
        post_mus, post_logvars = self.encode(x)
        zs = self.reparameterize(post_mus, post_logvars)
        
        # 获取先验参数
        prior_mus, prior_logvars = self.get_prior_params(zs)
        
        # 解码
        recon = self.decode(zs)
        
        # 计算层次KL散度
        kl_losses = []
        for i in range(self.n_levels):
            # 每层的KL: KL(q(z_i|z_{i+1}, x) || p(z_i|z_{i+1}))
            kl = 0.5 * torch.sum(
                prior_logvars[i] - post_logvars[i] +
                (post_logvars[i].exp() + (post_mus[i] - prior_mus[i]).pow(2)) / 
                prior_logvars[i].exp() - 1,
                dim=1
            )
            kl_losses.append(kl)
        
        total_kl = sum(kl_losses)
        
        # 重建损失
        recon_loss = F.binary_cross_entropy(recon, x, reduction='none').sum(dim=1)
        
        # 总损失
        loss = recon_loss + total_kl
        
        return {
            'loss': loss.mean(),
            'recon_loss': recon_loss.mean(),
            'kl_loss': total_kl.mean(),
            'kl_per_level': [kl.mean().item() for kl in kl_losses],
            'recon': recon
        }

# ============================================================================
# 4. VQ-VAE 实现
# ============================================================================

class VectorQuantizer(nn.Module):
    """向量量化层"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # 创建embedding表
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs):
        # inputs: [B, embedding_dim]
        
        # 计算距离
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # 找到最近的embedding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 量化
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view_as(inputs)
        
        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices.view(-1)

class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_embeddings):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 向量量化
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 编码
        z_e = self.encoder(x)
        
        # 量化
        z_q, vq_loss, encoding_indices = self.vq(z_e)
        
        # 解码
        recon = self.decoder(z_q)
        
        # 重建损失
        recon_loss = F.binary_cross_entropy(recon, x, reduction='none').sum(dim=1)
        
        # 总损失
        loss = recon_loss + vq_loss
        
        # 计算codebook使用率
        unique_encodings = len(torch.unique(encoding_indices))
        codebook_usage = unique_encodings / self.vq.num_embeddings
        
        return {
            'loss': loss.mean(),
            'recon_loss': recon_loss.mean(),
            'vq_loss': vq_loss,
            'codebook_usage': codebook_usage,
            'encoding_indices': encoding_indices,
            'recon': recon
        }

# ============================================================================
# 5. 训练脚本示例
# ============================================================================

def train_vae(model, dataloader, num_epochs=100, device='cuda'):
    """通用VAE训练函数"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            data = data.view(data.size(0), -1)  # flatten
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(data)
            loss = outputs['loss']
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 更新步数（KL Annealing需要）
            if hasattr(model, 'step'):
                model.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                
                # 打印额外信息
                if 'beta' in outputs:
                    print(f'  Beta: {outputs["beta"]:.4f}')
                if 'active_dims' in outputs:
                    print(f'  Active dims: {outputs["active_dims"]:.1f}')
                if 'codebook_usage' in outputs:
                    print(f'  Codebook usage: {outputs["codebook_usage"]:.2f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')

# ============================================================================
# 7. 完整使用示例 - 对比所有方法
# ============================================================================

if __name__ == "__main__":
    # 生成更真实的测试数据（模拟MNIST）
    torch.manual_seed(42)
    np.random.seed(42)
    
    def create_synthetic_data():
        """创建合成数据集，模拟不同类别"""
        n_samples = 2000
        n_features = 784
        n_classes = 10
        
        # 为每个类别创建不同的模式
        X = []
        y = []
        for class_id in range(n_classes):
            # 每个类别的基础模式
            base_pattern = torch.randn(n_features) * 0.5
            
            # 该类别的样本
            for _ in range(n_samples // n_classes):
                sample = base_pattern + torch.randn(n_features) * 0.3
                sample = torch.sigmoid(sample)  # 归一化到[0,1]
                X.append(sample)
                y.append(class_id)
        
        return torch.stack(X), torch.tensor(y)
    
    print("🔧 生成合成数据集...")
    X, y = create_synthetic_data()
    
    # 划分训练测试集
    train_size = int(0.8 * len(X))
    train_X, train_y = X[:train_size], y[:train_size]
    test_X, test_y = X[train_size:], y[train_size:]
    
    # 创建数据加载器
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"📊 数据集大小: 训练={len(train_X)}, 测试={len(test_X)}")
    
    # 定义所有要测试的模型
    models_to_test = {
        'KL Annealing VAE': KLAnnealingVAE(784, 400, 20),
        'Free Bits VAE': FreeBitsVAE(784, 400, 20, free_bits=2.0),
        'Hierarchical VAE': HierarchicalVAE(784, 400, [32, 16, 8]),
        'VQ-VAE': VQVAE(784, 400, 64, 512)
    }
    
    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用设备: {device}")
    
    # 训练每个模型并对比效果
    results = {}
    
    print("\n" + "="*80)
    print("🏁 开始对比训练不同VAE方法")
    print("="*80)
    
    for model_name, model in models_to_test.items():
        print(f"\n{'🚀 ' + model_name + ' ':=^60}")
        
        # 训练模型
        metrics = train_vae_with_monitoring(
            model=model,
            train_loader=train_loader, 
            test_loader=test_loader,
            num_epochs=15,  # 较少epoch便于快速测试
            device=device,
            model_name=model_name
        )
        
        results[model_name] = {
            'model': model,
            'metrics': metrics,
            'final_loss': metrics['total_loss'][-1],
            'final_recon': metrics['recon_loss'][-1],
            'final_kl': metrics['kl_loss'][-1]
        }
        
        print(f"✅ {model_name} 训练完成\n")
    
    # ============================================================================
    # 8. 最终对比分析
    # ============================================================================
    
    print("\n" + "="*80)
    print("📊 所有方法的最终对比分析")
    print("="*80)
    
    # 创建对比表格
    print(f"{'方法':<20} {'总损失':<10} {'重建损失':<10} {'KL损失':<10} {'评价'}")
    print("-" * 70)
    
    for name, result in results.items():
        final_loss = result['final_loss']
        final_recon = result['final_recon']  
        final_kl = result['final_kl']
        
        # 简单的效果评价
        if final_recon < 0.1 and final_kl < 5.0:
            evaluation = "🟢 优秀"
        elif final_recon < 0.2 and final_kl < 10.0:
            evaluation = "🟡 良好"  
        else:
            evaluation = "🔴 需改进"
            
        print(f"{name:<20} {final_loss:<10.4f} {final_recon:<10.4f} {final_kl:<10.4f} {evaluation}")
    
    # 找出最佳方法
    best_method = min(results.keys(), key=lambda x: results[x]['final_loss'])
    print(f"\n🏆 综合表现最佳: {best_method}")
    
    # 详细分析最佳方法
    print(f"\n🔍 {best_method} 详细分析:")
    best_model = results[best_method]['model']
    test_data_sample = test_X[:100].to(device)
    check_posterior_collapse(best_model, test_data_sample, best_method)
    
    print("\n" + "="*80)
    print("🎯 训练建议总结")
    print("="*80)
    
    print("""
    根据训练结果，以下是选择建议:
    
    🥇 KL Annealing VAE:
       - 实现简单，效果稳定
       - 适合: 初学者，快速原型
       - 注意: 需要调节退火策略
    
    🥈 Free Bits VAE:
       - 超参数容易调节
       - 适合: 对维度利用率有要求
       - 注意: free_bits设置很重要
    
    🥉 Hierarchical VAE:
       - 表达能力强，适合复杂数据
       - 适合: 层次结构明显的数据
       - 注意: 计算复杂度较高
    
    🏅 VQ-VAE:
       - 根本解决posterior collapse
       - 适合: 对生成质量要求高
       - 注意: 需要调节codebook大小
    """)
    
    print("\n✨ 实验完成! 你可以根据结果选择最适合你数据的方法。")

