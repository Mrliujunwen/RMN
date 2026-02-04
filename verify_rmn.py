"""
验证 RMN 模型是否能正确实例化和运行
"""

import torch
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# 模拟 args 参数
class MockArgs:
    def __init__(self):
        self.num_class = 64
        self.way = 5
        self.shot = 5
        self.query = 15
        self.temperature = 0.2
        self.temperature_attn = 5.0
        
        # CCMR 配置
        self.ccmr_hidden_dim = 256
        self.ccmr_d_state = 16
        self.ccmr_n_layers = 2
        self.ccmr_dropout = 0.1
        self.cross_kernel = 5
        
        # MEFS配置
        self.num_experts = 3
        self.router_hidden = 128
        
        # 融合配置
        self.fusion_method = 'concat'


def test_model(model_class, model_name, args):
    """测试单个模型"""
    logger.info(f'\n{"="*60}')
    logger.info(f'测试 {model_name}')
    logger.info(f'{"="*60}')
    
    try:
        # 创建模型
        model = model_class(args, mode='encoder')
        logger.info(f'✓ 模型创建成功')
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'  总参数量: {total_params:,}')
        logger.info(f'  可训练参数: {trainable_params:,}')
        
        # 测试前向传播
        batch_size = 4
        x = torch.randn(batch_size, 3, 84, 84)
        
        with torch.no_grad():
            # 测试编码器模式
            features = model.encode(x, do_gap=False)
            logger.info(f'✓ 编码器前向传播成功')
            logger.info(f'  输入形状: {x.shape}')
            logger.info(f'  输出形状: {features.shape}')
            
            # 检查特征增强模块
            if hasattr(model, 'feature_enhancer') and model.feature_enhancer is not None:
                logger.info(f'✓ 特征增强模块已启用')
                if model_name == 'RMN-FixedWeight':
                    logger.info(f'  类型: MEFS（固定权重）')
                    logger.info(f'  专家数量: {args.num_experts}')
                elif model_name == 'RMN-SoftRouting':
                    logger.info(f'  类型: MEFS（软路由）')
                    logger.info(f'  专家数量: {args.num_experts}')
                    logger.info(f'  路由隐藏层: {args.router_hidden}')
            else:
                logger.info(f'  特征增强模块: 未启用（Base版本）')
        
        logger.info(f'✓ {model_name} 测试通过')
        return True
        
    except Exception as e:
        logger.error(f'✗ {model_name} 测试失败: {str(e)}')
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info('\n' + '='*60)
    logger.info('RMN (Relation Mamba Network) 集成测试')
    logger.info('验证模型是否能正确实例化和运行')
    logger.info('='*60)
    
    # 导入模型
    from models.rmn import RMN_Base, RMN_FixedWeight, RMN_SoftRouting
    
    # 创建参数
    args = MockArgs()
    
    # 测试 Base (基准)
    logger.info('\n[基准] 测试 RMN-Base（BSC + CCMR）')
    base_success = test_model(RMN_Base, 'RMN-Base', args)
    
    # 测试 FixedWeight (消融实验)
    logger.info('\n[消融实验] 测试 RMN-FixedWeight（MEFS固定权重 + BSC + CCMR）')
    fixed_success = test_model(RMN_FixedWeight, 'RMN-FixedWeight', args)
    
    # 测试 SoftRouting (完整版)
    logger.info('\n[完整版] 测试 RMN-SoftRouting（MEFS软路由 + BSC + CCMR）')
    soft_success = test_model(RMN_SoftRouting, 'RMN-SoftRouting', args)
    
    # 总结
    logger.info('\n' + '='*60)
    logger.info('测试总结')
    logger.info('='*60)
    
    if base_success and fixed_success and soft_success:
        logger.info('✓ 所有测试通过！')
        logger.info('\n实现状态:')
        logger.info('  1. ✓ Base: BSC + CCMR - 基准版本')
        logger.info('  2. ✓ FixedWeight: MEFS（固定权重）- 消融实验')
        logger.info('  3. ✓ SoftRouting: MEFS（软路由）- 完整方案')
        logger.info('\n核心模块:')
        logger.info('  - ✓ MEFS (Multi-Expert Feature Synthesizer): 多专家特征合成器')
        logger.info('  - ✓ BSC (Bidirectional Spatial Correspondence): 双向空间对应模块')
        logger.info('  - ✓ CCMR (Cross-Correlation Mamba Reasoner): 交叉相关Mamba推理器')
    else:
        logger.info('✗ 部分测试失败，请检查错误信息')
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
