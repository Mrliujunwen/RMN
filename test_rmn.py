"""
RMN (Relation Mamba Network) Testing Script
关系Mamba网络测试脚本

用法示例:
    # 测试训练好的模型
    python test_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 -load_path checkpoints/rmn_soft/max_acc.pth -model_version soft
    
    # 测试Base版本
    python test_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 -load_path checkpoints/rmn_base/max_acc.pth -model_version base
    
    # 测试FixedWeight版本
    python test_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 -load_path checkpoints/rmn_fixed/max_acc.pth -model_version fixed
"""

import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, set_seed, setup_run
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.rmn import RMN_Base, RMN_FixedWeight, RMN_SoftRouting, RMN

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate(model, loader, args, set='test'):
    """评估模型"""
    model.eval()

    loss_meter = Meter()
    acc_meter = Meter()

    label = torch.arange(args.way).repeat(args.query).cuda()
    k = args.way * args.shot

    tqdm_gen = tqdm.tqdm(loader)

    with torch.no_grad():
        for i, (data, train_labels) in enumerate(tqdm_gen, 1):
            data = data.cuda()

            # 特征编码
            model.module.mode = 'encoder'
            data = model(data)
            data_shot, data_query = data[:k], data[k:]

            # 推理
            model.module.mode = 'fusion'
            logits = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))

            loss = F.cross_entropy(logits, label)
            acc = compute_accuracy(logits, label)

            loss_meter.update(loss.item())
            acc_meter.update(acc)
            tqdm_gen.set_description(
                f'[{set:^5}] avg.loss:{loss_meter.avg():.4f} | '
                f'avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})'
            )

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def test_main(args):
    """主测试函数"""
    # 构建模型
    set_seed(args.seed)
    
    model_version = getattr(args, 'model_version', 'soft')
    if model_version == 'soft':
        model = RMN_SoftRouting(args).cuda()
        logger.info('[INFO] 使用 RMN-SoftRouting（完整版）')
    elif model_version == 'fixed':
        model = RMN_FixedWeight(args).cuda()
        logger.info('[INFO] 使用 RMN-FixedWeight（消融实验）')
    elif model_version == 'base':
        model = RMN_Base(args).cuda()
        logger.info('[INFO] 使用 RMN-Base（基准版本）')
    else:
        model = RMN(args).cuda()
        logger.info('[INFO] 使用 RMN（默认完整版）')
    
    model = nn.DataParallel(model, device_ids=args.device_ids)
    
    # 加载模型
    if hasattr(args, 'load_path') and args.load_path:
        logger.info(f'[INFO] 从 {args.load_path} 加载模型')
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['params'])
        logger.info(f'[INFO] 模型加载成功 (epoch: {checkpoint.get("epoch", "unknown")})')
    else:
        logger.error('[ERROR] 请提供模型路径 -load_path')
        return
    
    # 打印配置
    logger.info(f'\n[INFO] RMN 测试配置:')
    logger.info(f'  - 模型版本: {model_version}')
    logger.info(f'  - 融合方式: {getattr(args, "fusion_method", "concat")}')
    logger.info(f'  - 数据集: {args.dataset}')
    logger.info(f'  - {args.way}-way {args.shot}-shot')
    logger.info(f'  - 测试episodes: {args.test_episode}')
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'\n[INFO] 总参数量: {total_params:,}')

    # 测试集
    Dataset = dataset_builder(args)
    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(
        testset.label, 
        args.test_episode, 
        args.way, 
        args.shot + args.query
    )
    test_loader = DataLoader(
        dataset=testset, 
        batch_sampler=test_sampler, 
        num_workers=8, 
        pin_memory=True
    )
    test_loader = [x for x in test_loader]

    # 测试
    _, test_acc, test_ci = evaluate(model, test_loader, args, set='test')
    
    logger.info(f'\n[最终结果] 测试准确率: {test_acc:.3f} ± {test_ci:.3f}')
    
    return test_acc, test_ci


if __name__ == '__main__':
    args = setup_run(arg_mode='test')
    
    logger.info('\n' + '='*60)
    logger.info('RMN (Relation Mamba Network) 测试')
    logger.info('='*60)

    test_acc, test_ci = test_main(args)
    
    logger.info('\n' + '='*60)
    logger.info(f'测试完成: {test_acc:.3f} ± {test_ci:.3f}')
    logger.info('='*60)
