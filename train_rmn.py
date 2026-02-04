"""
RMN (Relation Mamba Network) Training Script
关系Mamba网络训练脚本

用法示例:
    # Base版本
    python train_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 -extra_dir rmn_base -model_version base
    
    # FixedWeight版本（消融实验）
    python train_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 -extra_dir rmn_fixed -model_version fixed
    
    # SoftRouting版本（完整版）
    python train_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 -extra_dir rmn_soft -model_version soft
    
    # 使用自适应融合
    python train_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 -extra_dir rmn_adaptive -fusion_method adaptive
    
融合方式:
    - concat: 拼接两个分支的logits后通过FC (默认，推荐)
    - adaptive: 自适应学习BSC和CCMR的权重
    - add: 直接相加取平均
"""

import os
import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import detect_grad_nan, compute_accuracy, set_seed, setup_run
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.rmn import RMN_Base, RMN_FixedWeight, RMN_SoftRouting, RMN

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train(epoch, model, loader, optimizer, args=None):
    """训练一个epoch"""
    model.train()

    train_loader = loader['train_loader']
    train_loader_aux = loader['train_loader_aux']

    label = torch.arange(args.way).repeat(args.query).cuda()

    loss_meter = Meter()
    acc_meter = Meter()

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(train_loader)

    for i, ((data, train_labels), (data_aux, train_labels_aux)) in enumerate(zip(tqdm_gen, train_loader_aux), 1):

        data, train_labels = data.cuda(), train_labels.cuda()
        data_aux, train_labels_aux = data_aux.cuda(), train_labels_aux.cuda()

        # Stage 1: 特征编码
        model.module.mode = 'encoder'
        data = model(data)
        data_aux = model(data_aux)

        # Stage 2: 融合推理
        model.module.mode = 'fusion'
        data_shot, data_query = data[:k], data[k:]
        logits, absolute_logits = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))
        
        # 计算损失
        epi_loss = F.cross_entropy(logits, label)
        absolute_loss = F.cross_entropy(absolute_logits, train_labels[k:])

        # 辅助任务损失
        model.module.mode = 'fc'
        logits_aux = model(data_aux)
        loss_aux = F.cross_entropy(logits_aux, train_labels_aux)
        loss_aux = loss_aux + absolute_loss

        # 总损失
        loss = args.lamb * epi_loss + loss_aux
        acc = compute_accuracy(logits, label)

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(
            f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | '
            f'avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})'
        )

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        detect_grad_nan(model)
        optimizer.step()
        optimizer.zero_grad()

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def evaluate(epoch, model, loader, args, set='val'):
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
                f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | '
                f'avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})'
            )

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def train_main(args):
    """主训练函数"""
    Dataset = dataset_builder(args)

    # 训练集
    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(
        trainset.label, 
        len(trainset.data) // args.batch, 
        args.way, 
        args.shot + args.query
    )
    train_loader = DataLoader(
        dataset=trainset, 
        batch_sampler=train_sampler, 
        num_workers=8, 
        pin_memory=True
    )

    trainset_aux = Dataset('train', args)
    train_loader_aux = DataLoader(
        dataset=trainset_aux, 
        batch_size=args.batch, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True
    )

    train_loaders = {'train_loader': train_loader, 'train_loader_aux': train_loader_aux}

    # 验证集
    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(
        valset.label, 
        args.val_episode, 
        args.way, 
        args.shot + args.query
    )
    val_loader = DataLoader(
        dataset=valset, 
        batch_sampler=val_sampler, 
        num_workers=8, 
        pin_memory=True
    )
    val_loader = [x for x in val_loader]

    # 构建模型
    set_seed(args.seed)
    
    model_version = getattr(args, 'model_version', 'soft')
    if model_version == 'soft':
        model = RMN_SoftRouting(args).cuda()
        logger.info('[INFO] 使用 RMN-SoftRouting（完整版）')
        logger.info('       MEFS（软路由）+ BSC + CCMR')
    elif model_version == 'fixed':
        model = RMN_FixedWeight(args).cuda()
        logger.info('[INFO] 使用 RMN-FixedWeight（消融实验）')
        logger.info('       MEFS（固定权重）+ BSC + CCMR')
    elif model_version == 'base':
        model = RMN_Base(args).cuda()
        logger.info('[INFO] 使用 RMN-Base（基准版本）')
        logger.info('       BSC + CCMR（不使用MEFS）')
    else:
        model = RMN(args).cuda()
        logger.info('[INFO] 使用 RMN（默认完整版）')
    
    model = nn.DataParallel(model, device_ids=args.device_ids)
    
    # 打印配置
    logger.info(f'\n[INFO] RMN 配置:')
    logger.info(f'  - 模型版本: {model_version}')
    logger.info(f'  - 融合方式: {getattr(args, "fusion_method", "concat")}')
    logger.info(f'  - CCMR隐藏维度: {getattr(args, "ccmr_hidden_dim", 256)}')
    logger.info(f'  - CCMR状态维度: {getattr(args, "ccmr_d_state", 16)}')
    logger.info(f'  - CCMR层数: {getattr(args, "ccmr_n_layers", 2)}')
    logger.info(f'  - BSC温度: {getattr(args, "temperature_attn", 5.0)}')
    
    # FixedWeight和SoftRouting特殊参数
    if model_version in ['fixed', 'soft']:
        logger.info(f'  - MEFS: 启用')
        logger.info(f'  - 专家数量: {getattr(args, "num_experts", 3)}')
        if model_version == 'soft':
            logger.info(f'  - 软路由: 启用')
            logger.info(f'  - 路由隐藏维度: {getattr(args, "router_hidden", 128)}')
        else:
            logger.info(f'  - 软路由: 禁用（固定权重）')
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'\n[INFO] 总参数量: {total_params:,}')
    logger.info(f'[INFO] 可训练参数: {trainable_params:,}')

    # 优化器
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.9, 
        nesterov=True, 
        weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=args.milestones, 
        gamma=args.gamma
    )

    max_acc, max_epoch = 0.0, 0
    set_seed(args.seed)

    for epoch in range(1, args.max_epoch + 1):
        start_time = time.time()

        train_loss, train_acc, _ = train(epoch, model, train_loaders, optimizer, args)
        val_loss, val_acc, _ = evaluate(epoch, model, val_loader, args, set='val')

        if val_acc > max_acc:
            logger.info(f'[ log ] *********发现更好的模型 ({val_acc:.3f}) *********')
            max_acc, max_epoch = val_acc, epoch
            torch.save(
                dict(params=model.state_dict(), epoch=epoch), 
                os.path.join(args.save_path, 'max_acc.pth')
            )
            torch.save(
                optimizer.state_dict(), 
                os.path.join(args.save_path, 'optimizer_max_acc.pth')
            )

        if args.save_all:
            torch.save(
                dict(params=model.state_dict(), epoch=epoch), 
                os.path.join(args.save_path, f'epoch_{epoch}.pth')
            )

        epoch_time = time.time() - start_time
        logger.info(f'[ log ] 保存至 {args.save_path}')
        logger.info(f'[ log ] 预计剩余时间 {(args.max_epoch - epoch) / 3600. * epoch_time:.2f} 小时\n')

        lr_scheduler.step()

    logger.info(f'\n[ log ] 训练完成。最佳epoch: {max_epoch}, 最佳验证准确率: {max_acc:.3f}')
    return model, max_acc


def test_main(model, args):
    """测试函数"""
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
    _, test_acc, test_ci = evaluate(0, model, test_loader, args, set='test')
    
    return test_acc, test_ci


if __name__ == '__main__':
    args = setup_run(arg_mode='train')
    
    logger.info('\n' + '='*60)
    logger.info('RMN (Relation Mamba Network) 训练')
    logger.info('='*60)

    model, max_val_acc = train_main(args)
    test_acc, test_ci = test_main(model, args)
    
    logger.info(f'\n[最终结果] 测试准确率: {test_acc:.3f} ± {test_ci:.3f}')
    logger.info(f'[最终结果] 验证准确率: {max_val_acc:.3f}')
