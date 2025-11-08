import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from model import Transformer  # 在model里把完整模型叫Transformer
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import argparse

# 1. 加载并预处理数据集（文本生成类任务，Tiny Shakespeare）
def load_and_preprocess_data():
    dataset = load_dataset('tiny_shakespeare')
    train_dataset = dataset['train']['text'][0]  # 简单取训练集，实际要处理成序列
    src = torch.randint(0, 1000, (32, 50))  # 输入序列，batch_size=32, seq_len=50
    tgt = torch.randint(0, 1000, (32, 50))  # 目标序列
    return src, tgt

# 添加权重初始化
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        try:
            nn.init.xavier_uniform_(m.weight)
        except Exception:
            pass
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        try:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        except Exception:
            pass

# 统计模型参数量
def count_parameters(model):
    """统计模型参数量"""
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 按模块统计参数
    print("\nParameters by component:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,} parameters")
    
    # 总参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # 计算模型大小（MB）
    model_size = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024 / 1024
    print(f"Model Size: {model_size:.2f}MB")
    
    return total_params

# 2. 初始化模型、优化器、损失函数
def init_components(d_model=128, num_heads=4, num_layers=2, vocab_size=1000, device='cpu'):
    model = Transformer(d_model, num_heads, num_layers, vocab_size)
    model.apply(init_weights)
    model = model.to(device)
    
    # 打印模型结构和参数统计
    print("\nModel Architecture:")
    print(model)
    total_params = count_parameters(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # padding_idx
    return model, optimizer, scheduler, criterion

# 3. 训练循环
def train(model, optimizer, scheduler, criterion, src, tgt, epochs=10, vocab_size=1000):
    model.train()
    loss_history = []
    best_loss = float('inf')
    # 训练前的基本检查
    if tgt.max() >= vocab_size:
        raise ValueError(f"Found target id >= vocab_size ({int(tgt.max())} >= {vocab_size}). This will cause CrossEntropy error.")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])  # tgt输入去掉最后一个token
        logits = output.reshape(-1, output.size(-1))
        targets = tgt[:, 1:].reshape(-1)
        # 计算 loss 并检查
        loss = criterion(logits, targets)

        # 检查 loss/输出是否为 NaN/Inf，打印诊断并保存模型快照
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Epoch {epoch}] Loss is NaN/Inf. Dumping diagnostics...")
            print("Targets: min %d max %d" % (int(targets.min()), int(targets.max())))
            # 保存模型与小批量输入用于复现
            try:
                torch.save({
                    'model_state': model.state_dict(),
                    'src_sample': src,
                    'tgt_sample': tgt,
                }, f'nan_debug_epoch{epoch}.pth')
                print("Saved nan_debug_epoch{epoch}.pth")
            except Exception as e:
                print("Failed to save debug snapshot:", e)
            raise RuntimeError("Loss became NaN/Inf")

        # 反向传播（捕获异常并启用 anomaly detection 以便定位）
        try:
            loss.backward()
        except RuntimeError as e:
            print("Backward failed with error:", e)
            print("Enabling autograd anomaly detection and retrying one step to get traceback...")
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
        
        # 裁剪梯度并检查梯度是否包含 NaN/Inf
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        for name, p in model.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                print(f"NaN/Inf found in gradients of parameter: {name}")
                # 保存状态并抛出错误以便排查
                torch.save({'model_state': model.state_dict()}, 'nan_grad_debug.pth')
                raise RuntimeError(f"NaN/Inf in grad {name}")

        optimizer.step()

        # 更新学习率调度器
        scheduler.step(loss)
        
        # 记录损失
        current_loss = loss.item()
        loss_history.append(current_loss)
        
        # 保存最佳模型
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), 'best_model.pth')

        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return loss_history

# 4. 添加模型保存函数
def save_model(model, save_path='model.pth'):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# 5. 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
    
    # 加载数据
    src, tgt = load_and_preprocess_data()
    src = src.to(device)
    tgt = tgt.to(device)
    
    # 初始化组件
    model, optimizer, scheduler, criterion = init_components(device=device)
    # 训练模型
    loss_history = train(model, optimizer, scheduler, criterion, src, tgt, epochs=100, vocab_size=1000)
    
    # 保存模型
    save_model(model, 'transformer_model.pth')
    
    # 绘制损失曲线
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('./results/loss_curve.png')  # 保存到results文件夹
    plt.close()

def set_seed(seed):
    """Set random seed for reproducibility"""
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 开始训练
    main()