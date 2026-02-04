"""
CCMR (Cross-Correlation Mamba Reasoner)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ============================================================================
# SS2D: Selective Scan 2D - 2D空间选择性扫描模块
# ============================================================================

class SS2D(nn.Module):
    """
    Selective Scan 2D (SS2D) - VMamba风格的2D选择性扫描
    
    核心思想: 将2D特征图展开为4个方向的序列，分别进行选择性扫描，
    然后融合得到具有全局感受野的特征。
    
    四个扫描方向:
    1. 左上 → 右下 (row-major)
    2. 右下 → 左上 (反向)
    3. 左下 → 右上 (column-major)
    4. 右上 → 左下 (反向)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_model: 输入通道数
            d_state: SSM状态维度
            d_conv: 深度卷积核大小
            expand: 通道扩展因子
            dropout: Dropout比率
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # 输入投影: d_model -> d_inner * 2 (x 和 gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 2D深度可分离卷积 (用于局部特征提取)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_inner,
            bias=True
        )
        
        # SSM参数投影 (为4个方向共享基础参数，但有方向特定的投影)
        self.x_proj = nn.Linear(self.d_inner, (d_state * 2 + 1) * 4, bias=False)
        
        # dt投影 (4个方向)
        self.dt_projs = nn.ModuleList([
            nn.Linear(1, self.d_inner, bias=True)
            for _ in range(4)
        ])
        
        # A参数 (4个方向共享)
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            'n -> d n',
            d=self.d_inner
        )
        self.A_log = nn.Parameter(torch.log(A))
        
        # D参数 (skip connection, 4个方向)
        self.Ds = nn.Parameter(torch.ones(4, self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # LayerNorm
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def selective_scan_1d(self, x, delta, A, B, C, D):
        """
        1D选择性扫描 (用于每个方向)
        
        Args:
            x: (B, L, D)
            delta: (B, L, D)
            A: (D, N)
            B: (B, L, N)
            C: (B, L, N)
            D: (D,)
        """
        batch_size, seq_len, d_inner = x.shape
        n_state = A.shape[1]
        
        # 离散化
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)
        
        # 初始化状态
        h = torch.zeros(batch_size, d_inner, n_state, device=x.device, dtype=x.dtype)
        
        ys = []
        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t].unsqueeze(-1)
            y = torch.einsum('bdn,bln->bd', h, C[:, t:t+1])
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        y = y + x * D
        
        return y
    
    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, H, W, C) 或 (B, C, H, W)
        
        Returns:
            输出特征图，形状与输入相同
        """
        # 确保输入格式为 (B, H, W, C)
        if x.dim() == 4 and x.shape[1] == self.d_model:
            # 输入是 (B, C, H, W)
            x = rearrange(x, 'b c h w -> b h w c')
            channel_first = True
        else:
            channel_first = False
            
        B, H, W, C = x.shape
        L = H * W
        
        # 残差
        residual = x
        x = self.norm(x)
        
        # 输入投影
        xz = self.in_proj(x)  # (B, H, W, d_inner*2)
        x_proj, z = xz.chunk(2, dim=-1)  # 各自 (B, H, W, d_inner)
        
        # 2D卷积
        x_conv = rearrange(x_proj, 'b h w c -> b c h w')
        x_conv = self.conv2d(x_conv)
        x_conv = F.silu(x_conv)
        x_conv = rearrange(x_conv, 'b c h w -> b h w c')
        
        # 生成4个方向的序列
        # 方向1: 左上→右下 (行优先)
        x_1 = rearrange(x_conv, 'b h w c -> b (h w) c')
        # 方向2: 右下→左上 (反向)
        x_2 = torch.flip(x_1, dims=[1])
        # 方向3: 左下→右上 (列优先)
        x_3 = rearrange(x_conv, 'b h w c -> b (w h) c')
        # 方向4: 右上→左下 (反向)
        x_4 = torch.flip(x_3, dims=[1])
        
        xs = [x_1, x_2, x_3, x_4]
        
        # SSM参数
        x_flat = rearrange(x_conv, 'b h w c -> b (h w) c')
        x_ssm = self.x_proj(x_flat)  # (B, L, (2N+1)*4)
        
        # 分割为4个方向
        ssm_params = x_ssm.chunk(4, dim=-1)  # 4 x (B, L, 2N+1)
        
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # 对每个方向进行选择性扫描
        ys = []
        for i, (x_dir, ssm_param) in enumerate(zip(xs, ssm_params)):
            delta, B_ssm, C_ssm = torch.split(
                ssm_param, [1, self.d_state, self.d_state], dim=-1
            )
            delta = F.softplus(self.dt_projs[i](delta))
            
            y_dir = self.selective_scan_1d(x_dir, delta, A, B_ssm, C_ssm, self.Ds[i])
            ys.append(y_dir)
        
        # 反转方向2和方向4，恢复原始顺序
        ys[1] = torch.flip(ys[1], dims=[1])
        ys[3] = torch.flip(ys[3], dims=[1])
        
        # 将方向3和方向4从列优先转回行优先
        ys[2] = rearrange(ys[2].view(B, W, H, -1), 'b w h c -> b (h w) c')
        ys[3] = rearrange(ys[3].view(B, W, H, -1), 'b w h c -> b (h w) c')
        
        # 融合4个方向 (求和)
        y = ys[0] + ys[1] + ys[2] + ys[3]
        
        # 重塑回2D
        y = rearrange(y, 'b (h w) c -> b h w c', h=H, w=W)
        
        # 门控
        z = F.silu(z)
        y = y * z
        
        # 输出投影
        y = self.out_proj(y)
        y = self.dropout(y)
        
        # 残差连接
        y = y + residual
        
        # 恢复原始格式
        if channel_first:
            y = rearrange(y, 'b h w c -> b c h w')
        
        return y


class SS2DBlock(nn.Module):
    """
    SS2D Block with FFN
    
    完整的SS2D块，包含选择性扫描和前馈网络
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # SS2D层
        self.ss2d = SS2D(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )
        
        # FFN
        mlp_hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 或 (B, H, W, C)
        """
        # SS2D
        x = self.ss2d(x)
        
        # FFN (需要转换格式)
        if x.dim() == 4:
            if x.shape[1] != x.shape[-1]:  # (B, C, H, W)
                x = rearrange(x, 'b c h w -> b h w c')
                x = x + self.ffn(x)
                x = rearrange(x, 'b h w c -> b c h w')
            else:  # (B, H, W, C)
                x = x + self.ffn(x)
        
        return x


# ============================================================================
# Spatial Relation SS2D - 空间关系SS2D模块
# ============================================================================

class SpatialRelationSS2D(nn.Module):
    """
    空间关系SS2D模块
    
    用于处理支持集和查询集之间的空间关系，
    利用SS2D的2D扫描能力捕捉空间结构信息
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        d_state: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        
        # 多层SS2D块
        self.ss2d_blocks = nn.ModuleList([
            SS2DBlock(
                d_model=hidden_channels,
                d_state=d_state,
                d_conv=3,
                expand=2,
                mlp_ratio=2.0,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
        )
        
    def forward(self, x):
        """
        Args:
            x: 特征图 (B, C, H, W)
        
        Returns:
            增强后的特征图 (B, hidden_channels, H, W)
        """
        # 输入投影
        x = self.input_proj(x)
        
        # SS2D处理
        for ss2d_block in self.ss2d_blocks:
            x = ss2d_block(x)
        
        # 输出投影
        x = self.output_proj(x)
        
        return x


# ============================================================================
# 原有的1D Mamba模块
# ============================================================================

class MambaBlock(nn.Module):
    """
    Mamba Block - 选择性状态空间模型
    
    实现S4选择性扫描机制，能够动态地选择性处理序列信息
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: 输入特征维度
            d_state: SSM状态维度
            d_conv: 因果卷积核大小
            expand: 扩展因子
            dropout: Dropout比率
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # 输入投影层
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 因果卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        
        # SSM参数投影
        # x -> (delta, B, C) 用于选择性扫描
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # delta投影 (用于计算步长)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # A参数 (离散化的连续系统矩阵)
        # 使用特殊初始化确保稳定性
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            'n -> d n',
            d=self.d_inner
        )
        self.A_log = nn.Parameter(torch.log(A))
        
        # D参数 (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影层
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # LayerNorm和Dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def selective_scan(self, x, delta, A, B, C, D):
        """
        选择性扫描算法 - Mamba的核心
        
        Args:
            x: 输入 (B, L, D)
            delta: 时间步长 (B, L, D)
            A: 状态转移矩阵 (D, N)
            B: 输入矩阵 (B, L, N)
            C: 输出矩阵 (B, L, N)
            D: skip connection (D,)
        
        Returns:
            输出序列 (B, L, D)
        """
        batch_size, seq_len, d_inner = x.shape
        n_state = A.shape[1]
        
        # 离散化 A 和 B
        # delta_A = exp(delta * A)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N)
        # delta_B = delta * B
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)
        
        # 初始化状态
        h = torch.zeros(batch_size, d_inner, n_state, device=x.device, dtype=x.dtype)
        
        # 存储输出
        ys = []
        
        # 顺序扫描
        for t in range(seq_len):
            # 状态更新: h_t = A * h_{t-1} + B * x_t
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t].unsqueeze(-1)
            # 输出: y_t = C * h_t
            y = torch.einsum('bdn,bln->bd', h, C[:, t:t+1])
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (B, L, D)
        
        # 添加skip connection
        y = y + x * D
        
        return y
    
    def forward(self, x):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
        
        Returns:
            输出张量 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 残差连接
        residual = x
        x = self.norm(x)
        
        # 输入投影 -> (x_proj, gate)
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)  # 各自 (B, L, d_inner)
        
        # 因果卷积
        x_conv = rearrange(x_proj, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # 因果: 截断未来
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)
        
        # SSM参数投影
        x_ssm = self.x_proj(x_conv)  # (B, L, 2N+1)
        delta, B, C = torch.split(
            x_ssm,
            [1, self.d_state, self.d_state],
            dim=-1
        )
        
        # delta变换
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        
        # 获取A矩阵
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # 选择性扫描
        y = self.selective_scan(x_conv, delta, A, B, C, self.D)
        
        # 门控
        z = F.silu(z)
        y = y * z
        
        # 输出投影
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y + residual


# ============================================================================
# Cluster-Guided Mamba - 聚类引导的Mamba扫描模块
# ============================================================================

class SoftKMeans(nn.Module):
    """
    可微分的软K-Means聚类
    
    用于对特征进行软聚类，输出每个样本属于各个聚类的权重
    """
    
    def __init__(self, feature_dim: int, n_clusters: int = 5, n_iters: int = 3, temperature: float = 1.0):
        """
        Args:
            feature_dim: 特征维度
            n_clusters: 聚类数量
            n_iters: 迭代次数
            temperature: 软分配温度（越小越接近硬分配）
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.temperature = temperature
        
        # 可学习的初始聚类中心
        self.init_centers = nn.Parameter(torch.randn(n_clusters, feature_dim) * 0.02)
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, N, C) 或 (N, C)
        
        Returns:
            centers: 聚类中心 (B, K, C) 或 (K, C)
            weights: 软分配权重 (B, N, K) 或 (N, K)
            importance: 每个样本的重要性分数 (B, N) 或 (N,)
        """
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True
        
        B, N, C = x.shape
        K = self.n_clusters
        
        # 初始化聚类中心
        centers = self.init_centers.unsqueeze(0).expand(B, -1, -1)  # (B, K, C)
        
        # 迭代更新
        for _ in range(self.n_iters):
            # E步: 计算软分配
            # 距离: (B, N, K)
            dist = torch.cdist(x, centers)  # (B, N, K)
            weights = F.softmax(-dist / self.temperature, dim=-1)  # (B, N, K)
            
            # M步: 更新聚类中心
            # centers = sum(weights * x) / sum(weights)
            weighted_sum = torch.einsum('bnk,bnc->bkc', weights, x)  # (B, K, C)
            weight_sum = weights.sum(dim=1, keepdim=True).transpose(1, 2)  # (B, K, 1)
            centers = weighted_sum / (weight_sum + 1e-8)
        
        # 计算最终的软分配权重
        dist = torch.cdist(x, centers)
        weights = F.softmax(-dist / self.temperature, dim=-1)  # (B, N, K)
        
        # 计算每个样本的重要性分数
        # 越接近聚类中心的样本重要性越高
        min_dist = dist.min(dim=-1)[0]  # (B, N)
        importance = F.softmax(-min_dist / self.temperature, dim=-1)  # (B, N)
        
        if squeeze_batch:
            centers = centers.squeeze(0)
            weights = weights.squeeze(0)
            importance = importance.squeeze(0)
        
        return centers, weights, importance


class ClusterGuidedMambaBlock(nn.Module):
    """
    聚类引导的Mamba Block
    
    核心创新: 使用聚类权重来调制Mamba的选择性扫描
    - 聚类权重影响delta（时间步长），高权重样本有更大的状态更新
    - 可以按照聚类顺序重新排列序列，实现分层扫描
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # 输入投影层
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 因果卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        
        # SSM参数投影
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # delta投影
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # 聚类权重调制层 (核心创新)
        self.cluster_modulation = nn.Sequential(
            nn.Linear(1, self.d_inner // 4),
            nn.ReLU(),
            nn.Linear(self.d_inner // 4, self.d_inner),
            nn.Sigmoid()
        )
        
        # A参数
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            'n -> d n',
            d=self.d_inner
        )
        self.A_log = nn.Parameter(torch.log(A))
        
        # D参数 (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影层
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # LayerNorm和Dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def cluster_weighted_scan(self, x, delta, A, B, C, D, importance):
        """
        聚类权重调制的选择性扫描
        
        Args:
            x: 输入 (B, L, D)
            delta: 时间步长 (B, L, D)
            A: 状态转移矩阵 (D, N)
            B: 输入矩阵 (B, L, N)
            C: 输出矩阵 (B, L, N)
            D: skip connection (D,)
            importance: 聚类重要性权重 (B, L)
        
        Returns:
            输出序列 (B, L, D)
        """
        batch_size, seq_len, d_inner = x.shape
        n_state = A.shape[1]
        
        # 用聚类权重调制delta
        # 高重要性的样本有更大的步长，在状态中留下更深的印记
        importance_mod = self.cluster_modulation(importance.unsqueeze(-1))  # (B, L, d_inner)
        delta = delta * (1 + importance_mod)  # 权重越高，delta越大
        
        # 离散化 A 和 B
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)
        
        # 初始化状态
        h = torch.zeros(batch_size, d_inner, n_state, device=x.device, dtype=x.dtype)
        
        # 存储输出
        ys = []
        
        # 顺序扫描
        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t].unsqueeze(-1)
            y = torch.einsum('bdn,bln->bd', h, C[:, t:t+1])
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        y = y + x * D
        
        return y
    
    def forward(self, x, importance=None):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
            importance: 聚类重要性权重 (batch_size, seq_len)，可选
        
        Returns:
            输出张量 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 如果没有提供importance，使用均匀权重
        if importance is None:
            importance = torch.ones(batch_size, seq_len, device=x.device) / seq_len
        
        residual = x
        x = self.norm(x)
        
        # 输入投影
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        # 因果卷积
        x_conv = rearrange(x_proj, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)
        
        # SSM参数投影
        x_ssm = self.x_proj(x_conv)
        delta, B, C = torch.split(
            x_ssm,
            [1, self.d_state, self.d_state],
            dim=-1
        )
        
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log.float())
        
        # 聚类权重调制的选择性扫描
        y = self.cluster_weighted_scan(x_conv, delta, A, B, C, self.D, importance)
        
        # 门控
        z = F.silu(z)
        y = y * z
        
        # 输出投影
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y + residual


class ClusterOrderedMamba(nn.Module):
    """
    聚类排序的Mamba扫描
    
    核心思想: 按照聚类结构重新组织扫描顺序
    1. 先扫描各个聚类的中心样本（原型）
    2. 再按聚类分组扫描其他样本
    3. 这样可以让Mamba先建立类别的整体印象，再细化到具体样本
    """
    
    def __init__(
        self,
        d_model: int,
        n_clusters: int = 5,
        d_state: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_clusters = n_clusters
        
        # 软聚类模块
        self.soft_kmeans = SoftKMeans(
            feature_dim=d_model,
            n_clusters=n_clusters,
            n_iters=3,
            temperature=0.5
        )
        
        # 聚类引导的Mamba层
        self.mamba_layers = nn.ModuleList([
            ClusterGuidedMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # 反向Mamba层（双向扫描）
        self.mamba_layers_reverse = nn.ModuleList([
            ClusterGuidedMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # 双向融合
        self.fusion = nn.Linear(d_model * 2, d_model)
        
        # 最终归一化
        self.final_norm = nn.LayerNorm(d_model)
        
    def cluster_reorder(self, x, cluster_weights):
        """
        按聚类权重重新排序序列
        
        Args:
            x: 输入序列 (B, N, C)
            cluster_weights: 聚类权重 (B, N, K)
        
        Returns:
            reordered_x: 重排序后的序列 (B, N, C)
            reorder_indices: 排序索引，用于恢复原始顺序
        """
        B, N, C = x.shape
        K = cluster_weights.shape[-1]
        
        # 获取每个样本最可能属于的聚类
        cluster_assignment = cluster_weights.argmax(dim=-1)  # (B, N)
        
        # 获取每个样本在其聚类内的重要性
        max_weights = cluster_weights.max(dim=-1)[0]  # (B, N)
        
        # 创建排序键: 聚类ID * 大数 + (1 - 重要性)
        # 这样会先按聚类分组，组内按重要性降序
        sort_keys = cluster_assignment.float() * 1000 + (1 - max_weights)
        
        # 排序
        reorder_indices = sort_keys.argsort(dim=1)  # (B, N)
        
        # 重排序
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, N)
        reordered_x = x[batch_indices, reorder_indices]
        
        return reordered_x, reorder_indices
    
    def restore_order(self, x, reorder_indices):
        """
        恢复原始顺序
        """
        B, N, C = x.shape
        
        # 创建逆排序索引
        inverse_indices = reorder_indices.argsort(dim=1)
        
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, N)
        restored_x = x[batch_indices, inverse_indices]
        
        return restored_x
    
    def forward(self, x):
        """
        Args:
            x: 输入序列 (B, N, C)
        
        Returns:
            output: 输出序列 (B, N, C)
            cluster_info: 聚类信息字典
        """
        B, N, C = x.shape
        
        # 1. 软聚类
        centers, cluster_weights, importance = self.soft_kmeans(x)
        
        # 2. 按聚类重排序
        reordered_x, reorder_indices = self.cluster_reorder(x, cluster_weights)
        
        # 重排序importance
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, N)
        reordered_importance = importance[batch_indices, reorder_indices]
        
        # 3. 正向聚类引导的Mamba扫描
        x_forward = reordered_x
        for mamba_layer in self.mamba_layers:
            x_forward = mamba_layer(x_forward, reordered_importance)
        
        # 4. 反向扫描
        x_backward = torch.flip(reordered_x, dims=[1])
        flipped_importance = torch.flip(reordered_importance, dims=[1])
        for mamba_layer in self.mamba_layers_reverse:
            x_backward = mamba_layer(x_backward, flipped_importance)
        x_backward = torch.flip(x_backward, dims=[1])
        
        # 5. 双向融合
        x_fused = torch.cat([x_forward, x_backward], dim=-1)
        x_fused = self.fusion(x_fused)
        
        # 6. 恢复原始顺序
        output = self.restore_order(x_fused, reorder_indices)
        
        # 7. 最终归一化
        output = self.final_norm(output)
        
        cluster_info = {
            'centers': centers,
            'cluster_weights': cluster_weights,
            'importance': importance,
        }
        
        return output, cluster_info


class ClusterGuidedRelationReasoner(nn.Module):
    """
    聚类引导的关系推理器
    
    结合软聚类和Mamba的选择性扫描机制:
    1. 对支持集特征进行软聚类，找到类别原型
    2. 根据聚类权重指导Mamba的扫描顺序和强度
    3. 更"典型"的样本在扫描中有更大的影响力
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        n_clusters: int = 5,
        d_state: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_clusters = n_clusters
        
        # 关系对投影
        self.relation_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, hidden_dim) * 0.02)
        
        # 聚类排序的Mamba
        self.cluster_mamba = ClusterOrderedMamba(
            d_model=hidden_dim,
            n_clusters=n_clusters,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout
        )
        
    def forward(self, support_features, query_features, way, shot):
        """
        Args:
            support_features: 支持集特征 (way*shot, C)
            query_features: 查询集特征 (num_query, C)
            way: 类别数
            shot: 每类样本数
        
        Returns:
            decision_vectors: 决策向量 (num_query, way*shot, hidden_dim)
            cluster_info: 聚类信息
        """
        num_query = query_features.shape[0]
        num_support = support_features.shape[0]
        
        # 构建关系对
        support_expanded = support_features.unsqueeze(0).expand(num_query, -1, -1)
        query_expanded = query_features.unsqueeze(1).expand(-1, num_support, -1)
        relation_pairs = torch.cat([support_expanded, query_expanded], dim=-1)
        
        # 投影
        relation_seq = self.relation_proj(relation_pairs)
        
        # 添加位置编码
        relation_seq = relation_seq + self.pos_embedding[:, :num_support, :]
        
        # 聚类引导的Mamba扫描
        decision_vectors, cluster_info = self.cluster_mamba(relation_seq)
        
        return decision_vectors, cluster_info


# ============================================================================
# 原有的关系序列编码器
# ============================================================================

class RelationSequenceEncoder(nn.Module):
    """
    关系序列编码器
    
    将支持集和查询集的关系对特征转化为有序序列，
    并添加可学习的位置编码
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int, max_seq_len: int = 100):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 关系对特征投影
        # 输入: 拼接的(support, query)特征
        self.relation_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 可学习的位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )
        
        # 类别编码 (用于区分不同类别的关系)
        self.class_embedding = nn.Embedding(20, hidden_dim)  # 最多支持20个类别
        
    def forward(self, support_features, query_features, way, shot):
        """
        Args:
            support_features: 支持集特征 (way*shot, C)
            query_features: 查询集特征 (num_query, C)
            way: 类别数
            shot: 每类样本数
        
        Returns:
            relation_seq: 关系序列 (num_query, way*shot, hidden_dim)
            class_labels: 类别标签 (way*shot,)
        """
        num_query = query_features.shape[0]
        num_support = support_features.shape[0]  # way * shot
        
        # 扩展并拼接形成关系对
        # support: (way*shot, C) -> (num_query, way*shot, C)
        support_expanded = support_features.unsqueeze(0).expand(num_query, -1, -1)
        # query: (num_query, C) -> (num_query, way*shot, C)
        query_expanded = query_features.unsqueeze(1).expand(-1, num_support, -1)
        
        # 拼接关系对
        relation_pairs = torch.cat([support_expanded, query_expanded], dim=-1)
        # (num_query, way*shot, 2C)
        
        # 投影到隐藏空间
        relation_seq = self.relation_proj(relation_pairs)
        # (num_query, way*shot, hidden_dim)
        
        # 添加位置编码
        relation_seq = relation_seq + self.pos_embedding[:, :num_support, :]
        
        # 添加类别编码
        # 注意: CategoriesSampler 采样的数据是按轮次排列的 (abcdabcdabcd)
        # 而不是按类别分组 (aaaabbbbcccc)
        # 所以 class_labels 应该用 .repeat(shot) 而不是 .repeat_interleave(shot)
        class_labels = torch.arange(way, device=support_features.device).repeat(shot)
        class_emb = self.class_embedding(class_labels)
        relation_seq = relation_seq + class_emb.unsqueeze(0)
        
        return relation_seq, class_labels


class MambaRelationReasoner(nn.Module):
    """
    Mamba关系推理器 - MR-Net的核心组件
    
    使用多层Mamba块对关系序列进行选择性扫描，
    实现高效的关系推理
    
    支持两种模式:
    - 普通模式: 标准的双向Mamba扫描
    - 聚类模式: 使用聚类权重引导Mamba扫描顺序和强度
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        d_state: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_cluster: bool = False,
        n_clusters: int = 5,
    ):
        """
        Args:
            feature_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            d_state: SSM状态维度
            n_layers: Mamba层数
            dropout: Dropout比率
            use_cluster: 是否使用聚类引导的Mamba扫描
            n_clusters: 聚类数量 (仅在use_cluster=True时有效)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_cluster = use_cluster
        
        # 关系序列编码器
        self.relation_encoder = RelationSequenceEncoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )
        
        if use_cluster:
            # ========== 聚类引导模式 ==========
            # 使用ClusterOrderedMamba进行聚类权重引导的扫描
            self.cluster_mamba = ClusterOrderedMamba(
                d_model=hidden_dim,
                n_clusters=n_clusters,
                d_state=d_state,
                n_layers=n_layers,
                dropout=dropout
            )
            # 融合层 (cluster_mamba已经是双向的)
            self.fusion = None
        else:
            # ========== 普通模式 ==========
            # 多层Mamba块
            self.mamba_layers = nn.ModuleList([
                MambaBlock(
                    d_model=hidden_dim,
                    d_state=d_state,
                    d_conv=4,
                    expand=2,
                    dropout=dropout
                )
                for _ in range(n_layers)
            ])
            
            # 双向融合: 正向 + 反向扫描
            self.bidirectional = True
            if self.bidirectional:
                self.mamba_layers_reverse = nn.ModuleList([
                    MambaBlock(
                        d_model=hidden_dim,
                        d_state=d_state,
                        d_conv=4,
                        expand=2,
                        dropout=dropout
                    )
                    for _ in range(n_layers)
                ])
                self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 最终的LayerNorm
        self.final_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, support_features, query_features, way, shot):
        """
        Args:
            support_features: 支持集特征 (way*shot, C)
            query_features: 查询集特征 (num_query, C)
            way: 类别数
            shot: 每类样本数
        
        Returns:
            decision_vectors: 决策向量 (num_query, way*shot, hidden_dim)
            class_labels: 类别标签
            cluster_info: 聚类信息 (仅在use_cluster=True时返回，否则为None)
        """
        # 构建关系序列
        relation_seq, class_labels = self.relation_encoder(
            support_features, query_features, way, shot
        )
        # relation_seq: (num_query, way*shot, hidden_dim)
        
        cluster_info = None
        
        if self.use_cluster:
            # ========== 聚类引导模式 ==========
            # 使用聚类权重引导Mamba扫描
            x, cluster_info = self.cluster_mamba(relation_seq)
            decision_vectors = self.final_norm(x)
        else:
            # ========== 普通模式 ==========
            # 正向Mamba扫描
            x_forward = relation_seq
            for mamba_layer in self.mamba_layers:
                x_forward = mamba_layer(x_forward)
            
            if self.bidirectional:
                # 反向Mamba扫描
                x_backward = torch.flip(relation_seq, dims=[1])
                for mamba_layer in self.mamba_layers_reverse:
                    x_backward = mamba_layer(x_backward)
                x_backward = torch.flip(x_backward, dims=[1])
                
                # 双向融合
                x = torch.cat([x_forward, x_backward], dim=-1)
                x = self.fusion(x)
            else:
                x = x_forward
            
            # 最终归一化
            decision_vectors = self.final_norm(x)
        
        return decision_vectors, class_labels, cluster_info


class RelationClassifier(nn.Module):
    """
    关系分类器
    
    将Mamba输出的决策向量转化为类别预测
    """
    
    def __init__(self, hidden_dim: int, temperature: float = 0.2):
        super().__init__()
        
        self.temperature = temperature
        
        # 决策向量到分数的投影
        self.score_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, decision_vectors, way, shot):
        """
        Args:
            decision_vectors: 决策向量 (num_query, way*shot, hidden_dim)
            way: 类别数
            shot: 每类样本数
        
        Returns:
            logits: 分类logits (num_query, way)
        """
        num_query = decision_vectors.shape[0]
        
        # 计算每个关系对的分数
        scores = self.score_proj(decision_vectors).squeeze(-1)
        # (num_query, way*shot)
        
        # 重塑为 (num_query, way, shot)
        # 注意: CategoriesSampler 采样的数据是按轮次排列的 (abcdabcdabcd)
        # 所以需要先 view(shot, way) 再 permute
        scores = scores.view(num_query, shot, way)
        scores = scores.permute(0, 2, 1)  # (num_query, way, shot)
        
        # 对每个类别的shot取平均
        class_scores = scores.mean(dim=-1)  # (num_query, way)
        
        # 应用温度缩放
        logits = class_scores / self.temperature
        
        return logits

