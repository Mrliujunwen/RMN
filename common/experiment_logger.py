"""
实验日志记录器
记录运行命令、参数和实验结果

用法:
    from common.experiment_logger import ExperimentLogger
    
    logger = ExperimentLogger(args)
    logger.log_start()  # 记录开始
    ... 训练 ...
    logger.log_result(test_acc, test_ci, val_acc)  # 记录结果
"""

import os
import json
import datetime
import sys


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, args, log_dir='experiment_logs'):
        self.args = args
        self.log_dir = log_dir
        self.start_time = None
        self.end_time = None
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 日志文件路径
        self.summary_file = os.path.join(log_dir, 'experiments_summary.txt')
        self.json_file = os.path.join(log_dir, 'experiments.json')
        
    def get_command(self):
        """获取运行命令"""
        return ' '.join(sys.argv)
    
    def get_experiment_id(self):
        """生成实验ID"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset = getattr(self.args, 'dataset', 'unknown')
        way = getattr(self.args, 'way', 5)
        shot = getattr(self.args, 'shot', 1)
        extra_dir = getattr(self.args, 'extra_dir', 'default')
        return f"{timestamp}_{dataset}_{way}w{shot}s_{extra_dir}"
    
    def log_start(self):
        """记录实验开始"""
        self.start_time = datetime.datetime.now()
        self.experiment_id = self.get_experiment_id()
        
        print(f"\n{'='*60}")
        print(f"[EXPERIMENT START] {self.experiment_id}")
        print(f"[COMMAND] {self.get_command()}")
        print(f"[TIME] {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
    def log_result(self, test_acc, test_ci, val_acc=None, extra_info=None):
        """
        记录实验结果
        
        Args:
            test_acc: 测试准确率
            test_ci: 置信区间
            val_acc: 验证集最高准确率
            extra_info: 额外信息字典
        """
        self.end_time = datetime.datetime.now()
        duration = self.end_time - self.start_time if self.start_time else None
        
        # 构建实验记录
        record = {
            'experiment_id': self.experiment_id if hasattr(self, 'experiment_id') else self.get_experiment_id(),
            'command': self.get_command(),
            'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else None,
            'end_time': self.end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration': str(duration) if duration else None,
            'test_acc': round(test_acc.item() if hasattr(test_acc, 'item') else test_acc, 3),
            'test_ci': round(test_ci.item() if hasattr(test_ci, 'item') else test_ci, 3),
            'val_acc': round(val_acc.item() if hasattr(val_acc, 'item') else val_acc, 3) if val_acc is not None else None,
            'args': self._args_to_dict(),
        }
        
        if extra_info:
            record['extra_info'] = extra_info
        
        # 保存到JSON文件
        self._save_json(record)
        
        # 保存到摘要文件
        self._save_summary(record)
        
        # 打印结果
        self._print_result(record)
        
        return record
    
    def _args_to_dict(self):
        """将args转换为字典"""
        args_dict = {}
        for key, value in vars(self.args).items():
            # 跳过不可序列化的对象
            try:
                json.dumps(value)
                args_dict[key] = value
            except (TypeError, ValueError):
                args_dict[key] = str(value)
        return args_dict
    
    def _save_json(self, record):
        """保存到JSON文件"""
        # 读取现有记录
        records = []
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r') as f:
                    records = json.load(f)
            except:
                records = []
        
        # 添加新记录
        records.append(record)
        
        # 保存
        with open(self.json_file, 'w') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    
    def _save_summary(self, record):
        """保存到摘要文件"""
        with open(self.summary_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"实验ID: {record['experiment_id']}\n")
            f.write(f"时间: {record['start_time']} -> {record['end_time']}\n")
            f.write(f"耗时: {record['duration']}\n")
            f.write(f"命令: {record['command']}\n")
            f.write(f"\n--- 关键参数 ---\n")
            
            # 关键参数
            key_params = ['dataset', 'way', 'shot', 'model_version', 'model_type',
                         'mamba_hidden_dim', 'mamba_n_layers', 'use_cluster', 
                         'fusion_method', 'max_epoch', 'lr']
            for param in key_params:
                if param in record['args']:
                    f.write(f"  {param}: {record['args'][param]}\n")
            
            f.write(f"\n--- 结果 ---\n")
            f.write(f"  验证集最高: {record['val_acc']:.3f}%\n" if record['val_acc'] else "")
            f.write(f"  测试准确率: {record['test_acc']:.3f} ± {record['test_ci']:.3f}%\n")
            f.write(f"{'='*80}\n")
    
    def _print_result(self, record):
        """打印结果"""
        print(f"\n{'='*60}")
        print(f"[EXPERIMENT COMPLETE] {record['experiment_id']}")
        print(f"[DURATION] {record['duration']}")
        print(f"[RESULT] Test: {record['test_acc']:.3f} ± {record['test_ci']:.3f}%")
        if record['val_acc']:
            print(f"[VAL BEST] {record['val_acc']:.3f}%")
        print(f"[LOG SAVED] {self.summary_file}")
        print(f"{'='*60}\n")


def print_experiment_history(log_dir='experiment_logs', n=10):
    """打印最近n条实验记录"""
    json_file = os.path.join(log_dir, 'experiments.json')
    
    if not os.path.exists(json_file):
        print("没有找到实验记录")
        return
    
    with open(json_file, 'r') as f:
        records = json.load(f)
    
    print(f"\n{'='*100}")
    print(f"{'实验历史 (最近 ' + str(min(n, len(records))) + ' 条)'}")
    print(f"{'='*100}")
    print(f"{'ID':<40} | {'Dataset':<12} | {'Way':<4} | {'Shot':<4} | {'Test Acc':<12} | {'Val Acc':<10}")
    print(f"{'-'*100}")
    
    for record in records[-n:]:
        args = record.get('args', {})
        print(f"{record['experiment_id']:<40} | "
              f"{args.get('dataset', '-'):<12} | "
              f"{args.get('way', '-'):<4} | "
              f"{args.get('shot', '-'):<4} | "
              f"{record['test_acc']:.3f}±{record['test_ci']:.3f} | "
              f"{record.get('val_acc', '-')}")
    
    print(f"{'='*100}\n")


def compare_experiments(log_dir='experiment_logs', filter_dataset=None):
    """比较实验结果，按测试准确率排序"""
    json_file = os.path.join(log_dir, 'experiments.json')
    
    if not os.path.exists(json_file):
        print("没有找到实验记录")
        return
    
    with open(json_file, 'r') as f:
        records = json.load(f)
    
    # 过滤数据集
    if filter_dataset:
        records = [r for r in records if r.get('args', {}).get('dataset') == filter_dataset]
    
    # 按测试准确率排序
    records.sort(key=lambda x: x.get('test_acc', 0), reverse=True)
    
    print(f"\n{'='*120}")
    print(f"实验对比 (按测试准确率排序)")
    if filter_dataset:
        print(f"数据集: {filter_dataset}")
    print(f"{'='*120}")
    print(f"{'Rank':<5} | {'Test Acc':<12} | {'Val Acc':<10} | {'Model':<15} | {'Extra Dir':<20} | {'Command':<50}")
    print(f"{'-'*120}")
    
    for i, record in enumerate(records[:20], 1):
        args = record.get('args', {})
        model = args.get('model_version', args.get('model_type', '-'))
        extra_dir = args.get('extra_dir', '-')[:20]
        cmd = record.get('command', '-')
        if len(cmd) > 50:
            cmd = cmd[:47] + '...'
        
        print(f"{i:<5} | "
              f"{record['test_acc']:.3f}±{record['test_ci']:.3f} | "
              f"{record.get('val_acc', '-'):<10} | "
              f"{model:<15} | "
              f"{extra_dir:<20} | "
              f"{cmd}")
    
    print(f"{'='*120}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='实验日志管理')
    parser.add_argument('-action', type=str, default='history', 
                        choices=['history', 'compare'],
                        help='操作: history (查看历史), compare (对比实验)')
    parser.add_argument('-n', type=int, default=10, help='显示记录数')
    parser.add_argument('-dataset', type=str, default=None, help='过滤数据集')
    args = parser.parse_args()
    
    if args.action == 'history':
        print_experiment_history(n=args.n)
    elif args.action == 'compare':
        compare_experiments(filter_dataset=args.dataset)

