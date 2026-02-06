"""
RMN (Relation Mamba Network)
关系Mamba网络：面向小样本学习的细粒度交叉相关建模与多专家特征合成

核心组件：
1. MEFS (Multi-Expert Feature Synthesizer): 多专家特征合成器
2. BSC (Bidirectional Spatial Correspondence): 双向空间对应模块
3. CCMR (Cross-Correlation Mamba Reasoner): 交叉相关Mamba推理器

架构:
                              ┌─ BSC分支 (4D空间匹配) → bsc_logits ─┐
    图像 → ResNet → MEFS →    │                                       │ → 融合 → 分类
                              └─ CCMR分支 (序列推理) → ccmr_logits ──┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import ResNet
from models.bsc import BSC
from models.mefs import MEFS_FixedWeight, MEFS_SoftRouting
from models.ccmr import MambaRelationReasoner, RelationClassifier


class CrossCorrelationComputation(nn.Module):
    """计算support和query之间的交叉相关"""
    
    def __init__(self, kernel_size=(5, 5), padding=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        
    def forward(self, support, query):
        """
        计算交叉相关
        
        Args:
            support: 支持集特征 (way*shot, C, H, W)
            query: 查询集特征 (num_query, C, H, W)
        
        Returns:
            cross_corr: 交叉相关图 (num_query, way*shot, H*W, K*K)
        """
        support = F.normalize(support, dim=1, p=2, eps=1e-8)
        query = F.normalize(query, dim=1, p=2, eps=1e-8)
        
        num_spt, C, H, W = support.shape
        num_qry = query.shape[0]
        K = self.kernel_size[0]
        
        # 对support展开邻域: (num_spt, C*K*K, H*W)
        support_unfold = self.unfold(support)
        support_unfold = support_unfold.view(num_spt, C, K*K, H*W)
        support_unfold = support_unfold.permute(0, 3, 1, 2)  # (num_spt, H*W, C, K*K)
        
        # query展平: (num_qry, H*W, C)
        query_flat = query.view(num_qry, C, H*W).permute(0, 2, 1)
        
        # 交叉相关: (num_qry, num_spt, H*W, K*K)
        query_exp = query_flat.unsqueeze(1).unsqueeze(-1)  # (num_qry, 1, H*W, C, 1)
        support_exp = support_unfold.unsqueeze(0)  # (1, num_spt, H*W, C, K*K)
        cross_corr = (query_exp * support_exp).sum(dim=3)
        
        return cross_corr


class CrossCorrMambaReasoner(nn.Module):
    """CCMR: Cross-Correlation Mamba Reasoner"""
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        d_state: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1,
        kernel_size: int = 5,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        
        # 交叉相关计算
        self.cross_corr = CrossCorrelationComputation(
            kernel_size=(kernel_size, kernel_size),
            padding=kernel_size // 2
        )
        
        # 交叉相关特征投影
        corr_feat_dim = kernel_size * kernel_size
        self.corr_proj = nn.Sequential(
            nn.Linear(corr_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # 空间位置编码
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, 25, hidden_dim) * 0.02)
        
        # Mamba层（双向）
        from models.ccmr import MambaBlock
        
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=hidden_dim, d_state=d_state, d_conv=4, expand=2, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.mamba_layers_reverse = nn.ModuleList([
            MambaBlock(d_model=hidden_dim, d_state=d_state, d_conv=4, expand=2, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 空间聚合
        self.spatial_agg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, support, query, way, shot):
        num_spt = support.shape[0]
        num_qry = query.shape[0]
        H, W = support.shape[2], support.shape[3]
        
        # 1. 计算交叉相关
        cross_corr = self.cross_corr(support, query)  # (num_qry, num_spt, H*W, K*K)
        
        # 2. 投影到hidden_dim
        corr_feat = self.corr_proj(cross_corr)  # (num_qry, num_spt, H*W, hidden)
        
        # 3. 添加空间位置编码
        corr_feat = corr_feat + self.spatial_pos[:, :, :H*W, :]
        
        # 4. 重塑为序列
        seq_len = H * W
        corr_seq = corr_feat.view(num_qry * num_spt, seq_len, -1)
        
        # 5. 双向Mamba扫描
        x_forward = corr_seq
        for mamba_layer in self.mamba_layers:
            x_forward = mamba_layer(x_forward)
        
        x_backward = torch.flip(corr_seq, dims=[1])
        for mamba_layer in self.mamba_layers_reverse:
            x_backward = mamba_layer(x_backward)
        x_backward = torch.flip(x_backward, dims=[1])
        
        x = torch.cat([x_forward, x_backward], dim=-1)
        x = self.fusion(x)
        
        # 6. 空间聚合
        attn_weights = self.spatial_agg(x).softmax(dim=1)
        x_pooled = (x * attn_weights).sum(dim=1)
        
        # 7. 重塑
        decision_vectors = x_pooled.view(num_qry, num_spt, -1)
        decision_vectors = self.final_norm(decision_vectors)
        
        class_labels = torch.arange(way, device=support.device).repeat(shot)
        return decision_vectors, class_labels


class CrossCorrClassifier(nn.Module):
    """Classifier for cross-correlation features"""
    
    def __init__(self, hidden_dim: int, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.score_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, decision_vectors, way, shot):
        num_query = decision_vectors.shape[0]
        scores = self.score_proj(decision_vectors).squeeze(-1)
        scores = scores.view(num_query, shot, way)
        scores = scores.permute(0, 2, 1)
        class_scores = scores.mean(dim=-1)
        return class_scores / self.temperature


class RMN_Base(nn.Module):
    """RMN Base: BSC + CCMR (without MEFS)"""
    
    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args
        
        # ==================== 共享骨干网络 ====================
        self.encoder = ResNet(args=args)
        self.encoder_dim = 640
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class)
        
        # Feature enhancer (not used in Base version)
        self.feature_enhancer = None
        
        # ==================== BSC分支 ====================
        self.bsc_module = BSC(kernel_sizes=[3, 3], planes=[16, 1])
        self.bsc_1x1 = nn.Sequential(
            nn.Conv2d(self.encoder_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # ==================== CCMR分支 ====================
        # 参数名使用 mamba_* 以兼容训练脚本
        mamba_hidden_dim = getattr(args, 'mamba_hidden_dim', 256)
        mamba_d_state = getattr(args, 'mamba_d_state', 16)
        mamba_n_layers = getattr(args, 'mamba_n_layers', 2)
        mamba_dropout = getattr(args, 'mamba_dropout', 0.1)
        cross_kernel = getattr(args, 'cross_kernel', 5)
        
        self.ccmr_reasoner = CrossCorrMambaReasoner(
            feature_dim=self.encoder_dim,
            hidden_dim=mamba_hidden_dim,
            d_state=mamba_d_state,
            n_layers=mamba_n_layers,
            dropout=mamba_dropout,
            kernel_size=cross_kernel,
        )
        
        self.ccmr_classifier = CrossCorrClassifier(
            hidden_dim=mamba_hidden_dim,
            temperature=getattr(args, 'temperature', 0.2)
        )
        
        # ==================== 融合模块 ====================
        self.fusion_method = getattr(args, 'fusion_method', 'concat')
        
        if self.fusion_method == 'adaptive':
            self.fusion_weight = nn.Sequential(
                nn.Linear(args.way * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Softmax(dim=-1)
            )
        elif self.fusion_method == 'concat':
            self.fusion_fc = nn.Linear(args.way * 2, args.way)
    
    def forward(self, input):
        if self.mode == 'fc':
            return self.fc_forward(input)
        elif self.mode == 'encoder':
            return self.encode(input, do_gap=False)
        elif self.mode == 'fusion':
            spt, qry = input
            return self.fusion_forward(spt, qry)
        elif self.mode == 'bsc':
            spt, qry = input
            return self.bsc_forward(spt, qry)
        elif self.mode == 'ccmr':
            spt, qry = input
            return self.ccmr_forward(spt, qry)
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
    
    def fc_forward(self, x):
        x = x.mean(dim=[-1, -2])
        return self.fc(x)
    
    def encode(self, x, do_gap=True):
        x = self.encoder(x)
        
        # Feature enhancement (implemented in subclasses)
        if self.feature_enhancer is not None:
            identity = x
            x, enhancer_info = self.feature_enhancer(x)
            x = x + identity
            x = F.relu(x, inplace=True)
        
        if do_gap:
            return F.adaptive_avg_pool2d(x, 1)
        else:
            return x
    
    def bsc_forward(self, spt, qry):
        """BSC branch forward"""
        spt = spt.squeeze(0)
        spt = self.normalize_feature(spt)
        qry = self.normalize_feature(qry)
        
        corr4d = self.get_4d_correlation_map(spt, qry)
        num_qry, way, H_s, W_s, H_q, W_q = corr4d.size()
        
        corr4d = self.bsc_module(corr4d.view(-1, 1, H_s, W_s, H_q, W_q))
        corr4d_s = corr4d.view(num_qry, way, H_s * W_s, H_q, W_q)
        corr4d_q = corr4d.view(num_qry, way, H_s, W_s, H_q * W_q)
        
        corr4d_s = self.gaussian_normalize(corr4d_s, dim=2)
        corr4d_q = self.gaussian_normalize(corr4d_q, dim=4)
        
        corr4d_s = F.softmax(corr4d_s / self.args.temperature_attn, dim=2)
        corr4d_s = corr4d_s.view(num_qry, way, H_s, W_s, H_q, W_q)
        corr4d_q = F.softmax(corr4d_q / self.args.temperature_attn, dim=4)
        corr4d_q = corr4d_q.view(num_qry, way, H_s, W_s, H_q, W_q)
        
        attn_s = corr4d_s.sum(dim=[4, 5])
        attn_q = corr4d_q.sum(dim=[2, 3])
        
        spt_attended = attn_s.unsqueeze(2) * spt.unsqueeze(0)
        qry_attended = attn_q.unsqueeze(2) * qry.unsqueeze(1)
        
        if self.args.shot > 1:
            spt_attended = spt_attended.view(num_qry, self.args.shot, self.args.way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.shot, self.args.way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)
        
        spt_attended_pooled = spt_attended.mean(dim=[-1, -2])
        qry_attended_pooled = qry_attended.mean(dim=[-1, -2])
        qry_pooled = qry.mean(dim=[-1, -2])
        
        similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)
        bsc_logits = similarity_matrix / self.args.temperature
        
        if self.training:
            return bsc_logits, self.fc(qry_pooled)
        else:
            return bsc_logits
    
    def ccmr_forward(self, spt, qry):
        """CCMR branch forward"""
        spt = spt.squeeze(0)
        
        decision_vectors, class_labels = self.ccmr_reasoner(
            spt, qry, way=self.args.way, shot=self.args.shot
        )
        
        ccmr_logits = self.ccmr_classifier(
            decision_vectors, way=self.args.way, shot=self.args.shot
        )
        
        if self.training:
            qry_pooled = qry.mean(dim=[-1, -2])
            return ccmr_logits, self.fc(qry_pooled)
        else:
            return ccmr_logits
    
    def fusion_forward(self, spt, qry):
        """融合BSC和CCMR的前向传播"""
        spt_orig = spt.squeeze(0)
        
        # ========== BSC分支 ==========
        spt_bsc = self.normalize_feature(spt_orig)
        qry_bsc = self.normalize_feature(qry)
        
        corr4d = self.get_4d_correlation_map(spt_bsc, qry_bsc)
        num_qry, way, H_s, W_s, H_q, W_q = corr4d.size()
        
        corr4d = self.bsc_module(corr4d.view(-1, 1, H_s, W_s, H_q, W_q))
        corr4d_s = corr4d.view(num_qry, way, H_s * W_s, H_q, W_q)
        corr4d_q = corr4d.view(num_qry, way, H_s, W_s, H_q * W_q)
        
        corr4d_s = self.gaussian_normalize(corr4d_s, dim=2)
        corr4d_q = self.gaussian_normalize(corr4d_q, dim=4)
        
        corr4d_s = F.softmax(corr4d_s / self.args.temperature_attn, dim=2)
        corr4d_s = corr4d_s.view(num_qry, way, H_s, W_s, H_q, W_q)
        corr4d_q = F.softmax(corr4d_q / self.args.temperature_attn, dim=4)
        corr4d_q = corr4d_q.view(num_qry, way, H_s, W_s, H_q, W_q)
        
        attn_s = corr4d_s.sum(dim=[4, 5])
        attn_q = corr4d_q.sum(dim=[2, 3])
        
        spt_attended = attn_s.unsqueeze(2) * spt_bsc.unsqueeze(0)
        qry_attended = attn_q.unsqueeze(2) * qry_bsc.unsqueeze(1)
        
        if self.args.shot > 1:
            spt_attended = spt_attended.view(num_qry, self.args.shot, self.args.way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.shot, self.args.way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)
        
        spt_attended_pooled = spt_attended.mean(dim=[-1, -2])
        qry_attended_pooled = qry_attended.mean(dim=[-1, -2])
        
        bsc_similarity = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)
        bsc_logits = bsc_similarity / self.args.temperature
        
        # ========== CCMR分支 ==========
        decision_vectors, class_labels = self.ccmr_reasoner(
            spt_orig, qry, way=self.args.way, shot=self.args.shot
        )
        ccmr_logits = self.ccmr_classifier(
            decision_vectors, way=self.args.way, shot=self.args.shot
        )
        
        # ========== 融合 ==========
        if self.fusion_method == 'adaptive':
            combined = torch.cat([bsc_logits, ccmr_logits], dim=-1)
            weights = self.fusion_weight(combined)
            fused_logits = weights[:, 0:1] * bsc_logits + weights[:, 1:2] * ccmr_logits
        elif self.fusion_method == 'concat':
            combined = torch.cat([bsc_logits, ccmr_logits], dim=-1)
            fused_logits = self.fusion_fc(combined)
        else:  # 'add'
            fused_logits = (bsc_logits + ccmr_logits) / 2
        
        if self.training:
            qry_pooled = qry.mean(dim=[-1, -2])
            return fused_logits, self.fc(qry_pooled)
        else:
            return fused_logits
    
    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        return torch.div(x - x_mean, torch.sqrt(x_var + eps))

    def get_4d_correlation_map(self, spt, qry):
        way = spt.shape[0]
        num_qry = qry.shape[0]
        spt = self.bsc_1x1(spt)
        qry = self.bsc_1x1(qry)
        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)
        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)
        return torch.einsum('qncij,qnckl->qnijkl', spt, qry)

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)


# ============================================================================
# RMN-FixedWeight: MEFS（固定权重）+ BSC + CCMR（消融实验）
# ============================================================================

class RMN_FixedWeight(RMN_Base):
    """RMN with Fixed-Weight MEFS"""
    
    def __init__(self, args, mode=None):
        # Properly initialize parent class
        super().__init__(args, mode)
        
        # Override with MEFS fixed weights
        self.feature_enhancer = MEFS_FixedWeight(
            in_channels=640,
            num_experts=getattr(args, 'num_experts', 3)
        )


# ============================================================================
# RMN-SoftRouting: MEFS（软路由）+ BSC + CCMR（完整版）
# ============================================================================

class RMN_SoftRouting(RMN_Base):
    """RMN with Soft-Routing MEFS (Recommended)"""
    
    def __init__(self, args, mode=None):
        # Properly initialize parent class
        super().__init__(args, mode)
        
        # Override with MEFS soft routing
        self.feature_enhancer = MEFS_SoftRouting(
            in_channels=640,
            num_experts=getattr(args, 'num_experts', 3),
            router_hidden=getattr(args, 'router_hidden', 128)
        )


# Default to SoftRouting version
RMN = RMN_SoftRouting
