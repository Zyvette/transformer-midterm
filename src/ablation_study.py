import torch
import torch.nn as nn
import torch.optim as optim
from train import load_and_preprocess_data, init_weights
from model import Transformer
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

def run_experiment(config):
    """运行单个实验配置"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    src, tgt = load_and_preprocess_data()
    src = src.to(device)
    tgt = tgt.to(device)
    
    # 初始化模型
    model = Transformer(
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        vocab_size=config['vocab_size'],
        dropout=config['dropout']
    )
    model.apply(init_weights)
    model = model.to(device)
    
    # 优化器设置
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 是否使用学习率调度器
    scheduler = None
    if config['use_scheduler']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 训练
    model.train()
    loss_history = []
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        logits = output.reshape(-1, output.size(-1))
        targets = tgt[:, 1:].reshape(-1)
        loss = criterion(logits, targets)
        
        loss.backward()
        
        # 是否使用梯度裁剪
        if config['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step(loss)
            
        loss_history.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Experiment: {config["name"]}, Epoch [{epoch+1}/{config["epochs"]}], Loss: {loss.item():.6f}')
    
    return loss_history

def ablation_study():
    """进行一系列消融实验"""
    # 基础配置
    base_config = {
        'name': 'baseline',
        'd_model': 128,
        'num_heads': 4,
        'num_layers': 2,
        'vocab_size': 1000,
        'dropout': 0.1,
        'learning_rate': 1e-3,
        'epochs': 100,
        'use_scheduler': True,
        'use_grad_clip': True
    }
    
    # 实验配置列表
    experiments = [
        dict(base_config, name='baseline'),  # 基准模型
        dict(base_config, name='no_scheduler', use_scheduler=False),  # 无学习率调度
        dict(base_config, name='no_grad_clip', use_grad_clip=False),  # 无梯度裁剪
        dict(base_config, name='high_dropout', dropout=0.3),  # 高dropout
        dict(base_config, name='more_heads', num_heads=8),  # 更多注意力头
        dict(base_config, name='more_layers', num_layers=4),  # 更多层数
    ]
    
    # 创建结果目录
    results_dir = './results/ablation_study'
    os.makedirs(results_dir, exist_ok=True)
    
    # 运行所有实验
    all_results = {}
    for config in experiments:
        print(f"\nRunning experiment: {config['name']}")
        loss_history = run_experiment(config)
        all_results[config['name']] = loss_history
        
        # 保存实验配置
        with open(f'{results_dir}/config_{config["name"]}.json', 'w') as f:
            json.dump(config, f, indent=4)
    
    # 绘制对比图
    plt.figure(figsize=(12, 8))
    for name, loss_history in all_results.items():
        plt.plot(loss_history, label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Ablation Study Results')
    plt.legend()
    plt.grid(True)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'{results_dir}/ablation_study_{timestamp}.png')
    plt.close()
    
    # 保存数值结果
    with open(f'{results_dir}/results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f)

if __name__ == '__main__':
    ablation_study()