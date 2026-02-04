"""
MEFS (Multi-Expert Feature Synthesizer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 基础专家模块（单个专家）
# ============================================================================

class LocalStructureEncoder(nn.Module):
    """Local structure encoder (single expert)"""
    def __init__(self, in_channels=640, mid_channels=64, out_channels=640,
                 stride=(1, 1, 1), bias=False):
        super(LocalStructureEncoder, self).__init__()
        
        self.mid_channels = mid_channels
        
        # 压缩通道
        self.conv1x1_in = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3D 卷积提取局部结构特征
        # 5x5邻域 -> 3x3卷积 -> 3x3 -> 3x3卷积 -> 1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, (1, 3, 3),
                     stride=stride, bias=bias, padding=(0, 0, 0)),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, (1, 3, 3),
                     stride=stride, bias=bias, padding=(0, 0, 0)),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 恢复通道
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        """
        输入: x [b, c, h, w, u, v] - 局部结构相关张量 (u=v=5)
        输出: [b, c_out, h, w] - 增强特征
        """
        b, c, h, w, u, v = x.shape
        x = x.view(b, c, h * w, u * v)
        
        # 降维
        x = self.conv1x1_in(x)  # [b, in_c, hw, 25] -> [b, mid_c, hw, 25]
        
        # 3D卷积处理
        mid_c = x.shape[1]
        x = x.view(b, mid_c, h * w, u, v)  # [b, mid_c, hw, 5, 5]
        x = self.conv1(x)  # [b, mid_c, hw, 3, 3]
        x = self.conv2(x)  # [b, mid_c, hw, 1, 1]
        
        # 恢复空间维度并升维
        x = x.view(b, mid_c, h, w)
        x = self.conv1x1_out(x)  # [b, mid_c, h, w] -> [b, out_c, h, w]
        
        return x


class LocalStructureComputation(nn.Module):
    """Compute local structure correlations"""
    def __init__(self, kernel_size=(5, 5), padding=2):
        super(LocalStructureComputation, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        输入: x [b, c, h, w]
        输出: [b, c, h, w, u, v] - 局部结构相关张量
        """
        b, c, h, w = x.shape
        
        # L2归一化
        x = self.relu(x)
        x = F.normalize(x, dim=1, p=2)
        identity = x
        
        # 展开邻域
        x = self.unfold(x)  # [b, c*u*v, h*w]
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)
        
        # 计算局部结构相关性（逐元素乘法）
        x = x * identity.unsqueeze(2).unsqueeze(2)
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()  # [b, c, h, w, u, v]
        
        return x


# ============================================================================
# 固定权重版本（消融实验）
# ============================================================================

class MEFS_FixedWeight(nn.Module):
    """MEFS with fixed weights (for ablation study)"""
    
    def __init__(self, in_channels=640, num_experts=3, expert_channels=None):
        """
        参数:
            in_channels: 输入通道数
            num_experts: 专家数量
            expert_channels: 每个专家的中间通道数，默认 [64, 64, 64]
        """
        super(MEFS_FixedWeight, self).__init__()
        
        self.num_experts = num_experts
        self.in_channels = in_channels
        
        # 默认专家配置：三个相同容量的专家，通过不同初始化学习不同模式
        if expert_channels is None:
            expert_channels = [64, 64, 64][:num_experts]
        
        logger.info(f'[MEFS_FixedWeight] 初始化 {num_experts} 个专家，mid_channels: {expert_channels}')
        
        # 局部结构计算（共享）- 5x5邻域
        self.local_structure = LocalStructureComputation(kernel_size=(5, 5), padding=2)
        
        # 多个专家（不同容量）
        self.experts = nn.ModuleList([
            LocalStructureEncoder(
                in_channels=in_channels,
                mid_channels=mid_c,
                out_channels=in_channels
            )
            for mid_c in expert_channels
        ])
        
        # Fixed weights (average)
        self.register_buffer('fixed_weights', 
                           torch.ones(num_experts) / num_experts)
    
    def forward(self, x):
        """
        Args:
            x: Input features [b, c, h, w]
        Returns:
            fused_features: [b, c, h, w]
            info: Debug info dict
        """
        local_corr = self.local_structure(x)
        
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(local_corr)
            expert_outputs.append(expert_out)
        
        expert_stack = torch.stack(expert_outputs, dim=1)
        weights = self.fixed_weights.view(1, -1, 1, 1, 1)
        
        fused_features = (expert_stack * weights).sum(dim=1)
        
        # Return info for analysis
        info = {
            'weights': self.fixed_weights.cpu().detach().numpy(),
            'expert_outputs': [e.detach() for e in expert_outputs],
            'fusion_method': 'fixed_weight'
        }
        
        return fused_features, info


# ============================================================================
# 软路由版本（完整版）
# ============================================================================

class SoftRouter(nn.Module):
    """Soft routing network for adaptive expert weighting"""
    
    def __init__(self, in_channels, num_experts, hidden_dim=128):
        """
        参数:
            in_channels: 输入特征通道数
            num_experts: 专家数量
            hidden_dim: 隐藏层维度
        """
        super(SoftRouter, self).__init__()
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 路由网络：特征 -> 专家权重
        self.router = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        """
        参数:
            x: 输入特征 [b, c, h, w]
        
        返回:
            weights: 专家权重 [b, num_experts]
        """
        b, c, h, w = x.shape
        
        global_feat = self.gap(x).view(b, c)
        weights = self.router(global_feat)
        
        return weights


class MEFS_SoftRouting(nn.Module):
    """MEFS with soft routing (Recommended)"""
    
    def __init__(self, in_channels=640, num_experts=3, expert_channels=None, router_hidden=128):
        """
        参数:
            in_channels: 输入通道数
            num_experts: 专家数量
            expert_channels: 每个专家的中间通道数
            router_hidden: 路由网络隐藏层维度
        """
        super(MEFS_SoftRouting, self).__init__()
        
        self.num_experts = num_experts
        self.in_channels = in_channels
        
        # 默认专家配置：三个相同容量的专家，通过不同初始化学习不同模式
        if expert_channels is None:
            expert_channels = [64, 64, 64][:num_experts]
        
        logger.info(f'[MEFS_SoftRouting] 初始化 {num_experts} 个专家，mid_channels: {expert_channels}')
        logger.info(f'[MEFS_SoftRouting] 软路由隐藏层维度: {router_hidden}')
        
        # 局部结构计算（共享）- 5x5邻域
        self.local_structure = LocalStructureComputation(kernel_size=(5, 5), padding=2)
        
        # 多个专家（不同容量）
        self.experts = nn.ModuleList([
            LocalStructureEncoder(
                in_channels=in_channels,
                mid_channels=mid_c,
                out_channels=in_channels
            )
            for mid_c in expert_channels
        ])
        
        # Soft routing network
        self.router = SoftRouter(
            in_channels=in_channels,
            num_experts=num_experts,
            hidden_dim=router_hidden
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features [b, c, h, w]
        Returns:
            fused_features: [b, c, h, w]
            info: Debug info dict (including routing weights)
        """
        b, c, h, w = x.shape
        
        routing_weights = self.router(x)
        local_corr = self.local_structure(x)
        
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(local_corr)
            expert_outputs.append(expert_out)
        
        expert_stack = torch.stack(expert_outputs, dim=1)
        weights = routing_weights.view(b, self.num_experts, 1, 1, 1)
        
        fused_features = (expert_stack * weights).sum(dim=1)
        
        # Return info for analysis
        info = {
            'weights': routing_weights.cpu().detach().numpy(),
            'expert_outputs': [e.detach() for e in expert_outputs],
            'fusion_method': 'soft_routing'
        }
        
        return fused_features, info


# ============================================================================
# 辅助函数
# ============================================================================

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    """
    测试代码
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 测试固定权重版本
    print('\n' + '='*60)
    print('测试 MEFS_FixedWeight（消融实验）')
    print('='*60)
    
    model_fixed = MEFS_FixedWeight(in_channels=640, num_experts=3)
    params_fixed = count_parameters(model_fixed)
    print(f'参数量: {params_fixed:,}')
    
    # 输入特征图
    x = torch.randn(4, 640, 5, 5)
    out_fixed, info_fixed = model_fixed(x)
    print(f'输入形状: {x.shape}')
    print(f'输出形状: {out_fixed.shape}')
    print(f'固定权重: {info_fixed["weights"]}')
    
    # 测试软路由版本
    print('\n' + '='*60)
    print('测试 MEFS_SoftRouting（完整版）')
    print('='*60)
    
    model_soft = MEFS_SoftRouting(in_channels=640, num_experts=3, router_hidden=128)
    params_soft = count_parameters(model_soft)
    print(f'参数量: {params_soft:,}')
    
    out_soft, info_soft = model_soft(x)
    print(f'输入形状: {x.shape}')
    print(f'输出形状: {out_soft.shape}')
    print(f'软路由权重（批次1）: {info_soft["weights"][0]}')
    print(f'软路由权重（批次2）: {info_soft["weights"][1]}')
    
    print('\n参数量对比:')
    print(f'  固定权重: {params_fixed:,}')
    print(f'  软路由: {params_soft:,}')
    print(f'  增加参数: {params_soft - params_fixed:,} ({(params_soft/params_fixed-1)*100:.2f}%)')
