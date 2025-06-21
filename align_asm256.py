import os
import torch
import click
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import signal
import matplotlib.pyplot as plt
import numpy as np
import sys
import asm2vec
from torch.optim.lr_scheduler import _LRScheduler
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 定义ASM嵌入维度映射层
class ASMEmbeddingMapper(nn.Module):
    def __init__(self, input_dim, output_dim, max_seq_len=1000, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # 只有当处理序列时才需要这些组件
        self.position_encoder = PositionalEncoding(input_dim, dropout, max_seq_len)
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=input_dim*4,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, x, seq_lens=None):
        # 检查是否是批处理的序列数据还是单个向量
        is_single_vector = x.dim() == 1 or (x.dim() == 2 and not seq_lens)
        
        if is_single_vector:
            # 如果是单个向量（平均池化后的结果），直接映射
            return self.linear(x)
        
        # 确保输入是3D: [batch_size, seq_len, feature_dim]
        if x.dim() == 2:  # [seq_len, feature_dim]
            x = x.unsqueeze(0)  # 添加批次维度 [1, seq_len, feature_dim]
        
        # 应用位置编码和Transformer
        x = self.position_encoder(x)
        x = self.transformer_encoder(x)
        
        # 池化处理
        if seq_lens and len(seq_lens) == x.size(0):
            # 使用实际序列长度进行池化
            pooled = []
            for i, length in enumerate(seq_lens):
                pooled.append(x[i, :length].mean(dim=0))
            x = torch.stack(pooled)
        else:
            # 全序列池化
            x = x.mean(dim=1)
        
        # 映射到目标维度
        return self.linear(x)

# 定义位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# 定义学习率预热调度器
class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热阶段
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        
        # 预热结束后，使用设定的调度器
        if not self.finished:
            self.after_scheduler.base_lrs = self.base_lrs
            self.after_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
            self.finished = True
        return self.after_scheduler.get_lr()
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self.last_epoch < self.warmup_epochs:
            # 在预热阶段设置学习率
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        else:
            # 预热后使用设定的调度器
            self.after_scheduler.step()

# 定义全局变量以便在信号处理函数中使用
global_vars = {
    'output_path': None,
    'asm_model': None,
    'src_model': None,
    'asm_mapper': None,  # 添加映射器到全局变量
}

# 定义InfoNCELoss损失函数
class InfoNCELoss(nn.Module):
    def __init__(self, temperature, margin):
        super().__init__()

    def forward(self, asm_emb, src_emb):
        # 检查输入维度
        if asm_emb.dim() != 2 or src_emb.dim() != 2:
            print(f"维度不正确: asm_emb.shape={asm_emb.shape}, src_emb.shape={src_emb.shape}")
            # 如果是1维张量，扩展为2维
            if asm_emb.dim() == 1:
                asm_emb = asm_emb.unsqueeze(0)
            if src_emb.dim() == 1:
                src_emb = src_emb.unsqueeze(0)
        
        # 检查是否包含NaN或Inf值
        if torch.isnan(asm_emb).any() or torch.isinf(asm_emb).any():
            print("警告: asm_emb包含NaN或Inf值，已替换为0")
            asm_emb = torch.nan_to_num(asm_emb, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.isnan(src_emb).any() or torch.isinf(src_emb).any():
            print("警告: src_emb包含NaN或Inf值，已替换为0")
            src_emb = torch.nan_to_num(src_emb, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 确保特征维度相同
        if asm_emb.size(1) != src_emb.size(1):
            print(f"特征维度不匹配: asm_emb={asm_emb.size(1)}, src_emb={src_emb.size(1)}")
            # 重塑为较小的特征维度
            min_dim = min(asm_emb.size(1), src_emb.size(1))
            asm_emb = asm_emb[:, :min_dim]
            src_emb = src_emb[:, :min_dim]
        
        # 确保数据类型一致
        if asm_emb.dtype != src_emb.dtype:
            print(f"数据类型不一致: asm_emb={asm_emb.dtype}, src_emb={src_emb.dtype}")
            # 统一为float32
            asm_emb = asm_emb.float()
            src_emb = src_emb.float()
            
        # 检查批量大小
        if asm_emb.size(0) != src_emb.size(0):
            print(f"批量大小不匹配: asm_emb={asm_emb.size(0)}, src_emb={src_emb.size(0)}")
            # 取较小的批量大小
            min_batch = min(asm_emb.size(0), src_emb.size(0))
            asm_emb = asm_emb[:min_batch]
            src_emb = src_emb[:min_batch]
        
        # 计算余弦相似度矩阵
        asm_emb_norm = nn.functional.normalize(asm_emb, p=2, dim=1)  # 归一化汇编嵌入
        src_emb_norm = nn.functional.normalize(src_emb, p=2, dim=1)  # 归一化源代码嵌入
        
        # 安全计算相似度矩阵
        try:
            similarity_matrix = torch.mm(asm_emb_norm, src_emb_norm.T)  # 计算余弦相似度
        except RuntimeError as e:
            print(f"矩阵乘法出错: {e}")
            print(f"asm_emb_norm.shape={asm_emb_norm.shape}, src_emb_norm.shape={src_emb_norm.shape}")
            # 尝试使用einsum作为替代
            similarity_matrix = torch.einsum('ik,jk->ij', asm_emb_norm, src_emb_norm)
        
        # 创建标识矩阵，用于区分正样本和负样本
        batch_size = similarity_matrix.size(0)
        mask = torch.eye(batch_size, device=asm_emb.device)
        
        # 使用交叉熵损失
        labels = torch.arange(batch_size).to(asm_emb.device)
        
        # 安全计算损失
        try:
            loss = nn.functional.cross_entropy(similarity_matrix, labels)
        except RuntimeError as e:
            print(f"计算损失出错: {e}")
            # 使用更稳定的实现
            logits = similarity_matrix / 0.1  # 使用温度参数使分布更平滑
            loss = nn.functional.cross_entropy(logits, labels)
        
        return loss, similarity_matrix

# 定义数据集类
class CodeDataset(Dataset):
    def __init__(self, asm_data, src_data):
        self.asm_data = asm_data
        self.src_data = src_data

    def __len__(self):
        return len(self.asm_data)

    def __getitem__(self, idx):
        return self.asm_data[idx], self.src_data[idx]

def read_data(src_dir, asm_dir):
    src_data = []
    asm_data = []

    src_files = os.listdir(src_dir)
    asm_files = os.listdir(asm_dir)

    for src_file in src_files:
        src_path = os.path.join(src_dir, src_file)
        asm_file = src_file.replace('.c', '.s')
        asm_path = os.path.join(asm_dir, asm_file)

        # 检查汇编文件是否存在
        if not os.path.exists(asm_path):
            print(f"Warning: {asm_path} does not exist. Skipping this file.")
            continue

        # 读取源代码和汇编代码
        with open(src_path, 'r', encoding='latin-1', errors='ignore') as f:
            src_data.append(f.read())
        with open(asm_path, 'r', encoding='latin-1', errors='ignore') as f:
            asm_data.append(f.read())

    return src_data, asm_data

def save_model(output_path, asm_model, src_model, asm_mapper=None, current_epoch=None, train_losses=None, val_losses=None, 
               mrr_history=None, top1_history=None, top3_history=None):
    """保存模型的函数，包括当前的epoch信息和损失历史"""
    save_dict = {
        'asm_model_state_dict': asm_model.state_dict(),
        'src_model_state_dict': src_model.state_dict(),
    }
    
    # 如果提供了ASM映射器，也保存它
    if asm_mapper is not None:
        save_dict['asm_mapper_state_dict'] = asm_mapper.state_dict()
    
    # 如果提供了当前epoch信息，也保存它
    if current_epoch is not None:
        save_dict['current_epoch'] = current_epoch
    
    # 保存训练和验证损失历史
    if train_losses is not None:
        save_dict['train_losses'] = train_losses
    if val_losses is not None:
        save_dict['val_losses'] = val_losses
    
    # 保存MRR历史
    if mrr_history is not None:
        save_dict['mrr_history'] = mrr_history
    
    # 保存Top-1和Top-3准确率历史
    if top1_history is not None:
        save_dict['top1_history'] = top1_history
    if top3_history is not None:
        save_dict['top3_history'] = top3_history
    
    torch.save(save_dict, output_path)
    print(f"模型已保存到 {output_path}.")

def signal_handler(sig, frame):
    """当捕获到中断信号时，保存当前模型"""
    print("\n捕获到中断信号 (Ctrl+C)，正在保存当前模型状态...")
    
    # 保存双模型和映射器
    save_model(
        global_vars['output_path'], 
        global_vars['asm_model'],
        global_vars['src_model'],
        global_vars['asm_mapper'],
        global_vars.get('current_epoch'),
        global_vars.get('train_losses'),
        global_vars.get('val_losses'),
        global_vars.get('mrr_history'),
        global_vars.get('top1_history'),
        global_vars.get('top3_history')
    )
    
    # 绘制所有指标曲线
    if (global_vars.get('train_losses') and global_vars.get('val_losses') and 
        global_vars.get('mrr_history') and global_vars.get('top1_history') and 
        global_vars.get('top3_history')):
        plot_all_metrics(
            global_vars['train_losses'], 
            global_vars['val_losses'],
            global_vars['mrr_history'],
            global_vars['top1_history'],
            global_vars['top3_history'],
            os.path.dirname(global_vars['output_path'])
        )
    
    print("模型已保存，程序退出。")
    exit(0)

def plot_all_metrics(train_losses, val_losses, mrr_history, top1_history, top3_history, save_dir):
    """将所有评估指标绘制在同一张图上"""
    # 创建一个包含3个子图的大图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 子图1：绘制损失曲线
    ax1.plot(train_losses, 'b-', label='train-loss')
    ax1.plot(val_losses, 'r-', label='val-loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True)
    
    # 子图2：绘制MRR曲线
    ax2.plot(mrr_history, 'g-', marker='o', label='MRR')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MRR')
    ax2.set_title('Mean Reciprocal Rank')
    ax2.set_ylim(0, 1.05)  # MRR范围在0到1之间
    ax2.legend()
    ax2.grid(True)
    
    # 子图3：绘制Top-1和Top-3准确率曲线
    ax3.plot(top1_history, 'c-', marker='^', label='Top-1acc')
    ax3.plot(top3_history, 'm-', marker='s', label='Top-3acc')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Top-N Accuracy')
    ax3.set_ylim(0, 1.05)  # 准确率范围在0到1之间
    ax3.legend()
    ax3.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    metrics_plot_path = os.path.join(save_dir, 'asm2vec_metrics.png')
    plt.savefig(metrics_plot_path)
    print(f"所有指标曲线已保存到 {metrics_plot_path}")

@click.command()
@click.option('-a', '--asm-dir', 'asm_dir', default='data/asm502', help='包含汇编文件的目录', required=True)
@click.option('-s', '--src-dir', 'src_dir', default='data/src', help='包含源代码文件的目录', required=True)
@click.option('-o', '--output', 'output_path', default='model_asm2vec5012.pt', show_default=True, help='模型保存路径')
@click.option('-e', '--epochs', default=200, show_default=True, help='训练轮数')
@click.option('-b', '--batch-size', 'batch_size', default=8, show_default=True, help='批量大小')
@click.option('-acc', '--accumulation-steps', 'accumulation_steps', default=4, show_default=True, help='梯度累积步数')
@click.option('-lr', '--learning-rate', 'lr', default=2e-5, show_default=True, help='学习率')
@click.option('-r', '--resume', is_flag=True, help='是否从上次保存的模型继续训练')
@click.option('-vs', '--val-split', 'val_split', default=0.2, show_default=True, help='验证集比例')
@click.option('-am', '--asm-model', 'asm_model_path', default='model420.pt', help='ASM2VEC预训练模型路径', required=True)
def train_model(asm_dir, src_dir, output_path, epochs, batch_size, lr, resume, val_split, accumulation_steps, asm_model_path):
    # 将参数存储在全局变量中，供信号处理函数使用
    global_vars['output_path'] = output_path
    
    # 注册信号处理器，捕获Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # 读取数据
    src_data, asm_data = read_data(src_dir, asm_dir)
    dataset = CodeDataset(asm_data, src_data)
    
    # 划分训练集和验证集
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                             generator=torch.Generator().manual_seed(42))
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"数据集大小: {dataset_size}, 训练集: {train_size}, 验证集: {val_size}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 设置起始epoch
    start_epoch = 0
    
    # 加载ASM2VEC预训练模型
    print(f"从 {asm_model_path} 加载ASM2VEC预训练模型...")
    asm_model, tokens = asm2vec.utils.load_model(asm_model_path, device=device)
    
    # 初始化CodeBERT模型用于处理源代码
    src_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    src_model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
    src_model.gradient_checkpointing_enable()
    
    # 创建ASM嵌入映射层 - 从ASM2VEC嵌入维度映射到CodeBERT嵌入维度
    asm_embedding_dim = asm_model.embeddings_r.embedding_dim  # 通常是400
    src_embedding_dim = src_model.config.hidden_size  # CodeBERT是768
    asm_mapper = ASMEmbeddingMapper(asm_embedding_dim, src_embedding_dim).to(device)
    print(f"创建ASM嵌入映射层: {asm_embedding_dim} -> {src_embedding_dim}")
    
    # 初始化损失历史记录和MRR历史
    train_losses = []
    val_losses = []
    mrr_history = []
    top1_history = []
    top3_history = []
    
    # 如果resume标志为True且模型文件存在，则加载模型
    if resume and os.path.exists(output_path):
        print(f"从 {output_path} 加载模型状态...")
        checkpoint = torch.load(output_path)
        
        # 加载源代码模型状态
        if 'src_model_state_dict' in checkpoint:
            src_model.load_state_dict(checkpoint['src_model_state_dict'])
        
        # 加载ASM映射器状态（如果存在）
        if 'asm_mapper_state_dict' in checkpoint:
            asm_mapper.load_state_dict(checkpoint['asm_mapper_state_dict'])
            print("加载了ASM映射器状态")
        else:
            print("未找到ASM映射器状态，使用新初始化的映射器")
        
        # 如果保存了当前epoch，则从该epoch继续训练
        if 'current_epoch' in checkpoint:
            start_epoch = checkpoint['current_epoch'] + 1
            print(f"继续从第 {start_epoch} 个epoch开始训练")
        
        # 加载损失历史记录
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            val_losses = checkpoint['val_losses']
        
        # 加载MRR历史记录
        if 'mrr_history' in checkpoint:
            mrr_history = checkpoint['mrr_history']
            
        # 加载Top-1和Top-3准确率历史
        if 'top1_history' in checkpoint:
            top1_history = checkpoint['top1_history']
        if 'top3_history' in checkpoint:
            top3_history = checkpoint['top3_history']

    # 将模型和历史记录存储在全局变量中
    global_vars['asm_model'] = asm_model
    global_vars['src_model'] = src_model
    global_vars['asm_mapper'] = asm_mapper
    global_vars['train_losses'] = train_losses
    global_vars['val_losses'] = val_losses
    global_vars['mrr_history'] = mrr_history
    global_vars['top1_history'] = top1_history
    global_vars['top3_history'] = top3_history

    # 定义损失和优化器
    criterion = InfoNCELoss(temperature=0.1, margin=0.5)
    
    # 优化源代码模型参数和ASM映射层参数，ASM2VEC保持固定
    optimizer = optim.AdamW(
        list(src_model.parameters()) + list(asm_mapper.parameters()),
        lr=lr, 
        weight_decay=0.01
    )

    # 学习率调整策略: 使用余弦退火并加入预热
    warmup_epochs = int(epochs * 0.1)  # 10%的轮数用于预热
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=lr/100)
    scheduler = LinearWarmupScheduler(optimizer, warmup_epochs=warmup_epochs, after_scheduler=main_scheduler)

    # 从设定的起始epoch开始训练
    for epoch in range(start_epoch, epochs):
        # 更新当前epoch
        global_vars['current_epoch'] = epoch
        
        # ================ 训练阶段 ================
        src_model.train()
        asm_mapper.train()
        asm_model.eval()  # ASM2VEC始终处于评估模式
        total_train_loss = 0
        optimizer.zero_grad()  # 在每个 epoch 开始时清零梯度
        
        # 用于梯度累积的计数器
        accumulation_counter = 0

        for batch_idx, (asm_code, src_code) in enumerate(tqdm(train_dataloader, desc=f'训练 Epoch {epoch+1}/{epochs}')):
            # 清理 CUDA 缓存
            if device == 'cuda':
                torch.cuda.empty_cache()

            # 处理汇编代码，使用ASM2VEC模型获取嵌入
            asm_embedding_list = []
            asm_seq_list = []  # 存储序列数据
            asm_seq_lens = []  # 存储序列长度
            
            for code in asm_code:
                # 将汇编代码转换为ASM2VEC能够处理的格式
                try:
                    # 创建Function对象并获取嵌入向量
                    fn = asm2vec.datatype.Function.load(code)
                    fn_tokens = fn.tokens()
                    
                    # 为每个token添加到词汇表
                    token_ids = []
                    for token in fn_tokens:
                        if token not in tokens.name_to_index:
                            tokens.add(token)
                        token_ids.append(tokens.name_to_index[token])
                    
                    # 使用embeddings_r获取token的嵌入，保留序列信息
                    with torch.no_grad():
                        if token_ids:
                            # 确保所有token_ids都在有效范围内
                            max_idx = asm_model.embeddings_r.num_embeddings - 1
                            valid_token_ids = [min(idx, max_idx) for idx in token_ids]
                            
                            try:
                                # 首先尝试在GPU上处理
                                token_tensor = torch.tensor(valid_token_ids, device=device)
                                token_embeddings = asm_model.embeddings_r(token_tensor)
                                
                                # 如果序列足够长，使用序列模式
                                if len(token_embeddings) > 5:  # 最小长度阈值
                                    if len(token_embeddings) > 1000:  # 截断过长序列
                                        token_embeddings = token_embeddings[:1000]
                                    asm_seq_list.append(token_embeddings)
                                    asm_seq_lens.append(len(token_embeddings))
                                else:
                                    # 对于短序列使用平均池化
                                    embedding = token_embeddings.mean(dim=0)
                                    asm_embedding_list.append(embedding)
                            except RuntimeError as e:
                                # 如果GPU出错，尝试在CPU上处理
                                print(f"GPU处理出错，转为CPU: {e}")
                                cpu_token_tensor = torch.tensor(valid_token_ids)
                                cpu_embeddings = asm_model.embeddings_r.cpu()(cpu_token_tensor)
                                embedding = cpu_embeddings.mean(dim=0).to(device)
                                asm_embedding_list.append(embedding)
                        else:
                            # 如果没有有效token，使用零向量
                            try:
                                embedding = torch.zeros(asm_model.embeddings_r.embedding_dim, device=device)
                            except RuntimeError:
                                # 如果GPU创建失败，使用CPU
                                embedding = torch.zeros(asm_model.embeddings_r.embedding_dim).to(device)
                            asm_embedding_list.append(embedding)
                except Exception as e:
                    print(f"处理汇编代码时出错: {e}")
                    # 出错时使用零向量
                    try:
                        embedding_size = asm_model.embeddings_r.embedding_dim
                        embedding = torch.zeros(embedding_size, device=device)
                    except RuntimeError:
                        # 如果GPU创建失败，使用CPU
                        embedding_size = asm_model.embeddings_r.embedding_dim
                        embedding = torch.zeros(embedding_size).to(device)
                    asm_embedding_list.append(embedding)
            
            # 处理嵌入
            if asm_seq_list and len(asm_seq_list) == len(asm_code):
                # 如果所有样本都是序列，使用填充批处理
                max_len = max(len(seq) for seq in asm_seq_list)
                padded_seqs = []
                for seq in asm_seq_list:
                    if len(seq) < max_len:
                        padding = torch.zeros(max_len - len(seq), seq.size(1), device=device)
                        padded_seq = torch.cat([seq, padding], dim=0)
                    else:
                        padded_seq = seq
                    padded_seqs.append(padded_seq)
                asm_embedding_batch = torch.stack(padded_seqs)
                # 使用映射层处理批次序列
                asm_embedding = asm_mapper(asm_embedding_batch, asm_seq_lens)
            elif asm_embedding_list:
                # 否则使用平均嵌入
                asm_embedding = torch.stack(asm_embedding_list)
                # 使用映射层将ASM嵌入映射到与源代码嵌入相同的维度
                asm_embedding = asm_mapper(asm_embedding)
            else:
                # 如果两者都为空，创建零张量批次
                asm_embedding = torch.zeros(len(asm_code), asm_model.embeddings_r.embedding_dim, device=device)
                asm_embedding = asm_mapper(asm_embedding)
            
            # 处理源代码，使用CodeBERT模型获取嵌入
            src_embedding_list = []
            for code in src_code:
                # 分批处理长代码
                max_length = 512  # 减小最大长度以节省内存
                tokens_bert = src_tokenizer(code, return_tensors='pt', truncation=True, padding=False, max_length=max_length)
                
                # 如果代码长度小于最大长度，直接处理
                if tokens_bert['input_ids'].size(1) <= max_length:
                    inputs = {k: v.to(device) for k, v in tokens_bert.items()}
                    outputs = src_model(**inputs)
                    # 使用[CLS]标记的嵌入作为代码表示
                    cls_embedding = outputs.last_hidden_state[:, 0, :]
                    
                else:
                    # 使用滑动窗口处理长代码
                    # print("代码长度超出设定的max_length，使用滑动窗口处理")
                    window_size = max_length - 50  # 窗口大小（留出一些重叠空间）
                    stride = window_size // 2  # 滑动步长（50%重叠）
                    
                    # 将代码分词
                    all_tokens = src_tokenizer(code, add_special_tokens=False)
                    input_ids = all_tokens['input_ids']
                    
                    # 存储每个窗口的嵌入向量
                    window_embeddings = []
                    
                    # 分窗口处理
                    for start_idx in range(0, len(input_ids), stride):
                        end_idx = min(start_idx + window_size, len(input_ids))
                        if end_idx - start_idx < 100:  # 跳过太小的窗口
                            continue
                            
                        # 为窗口添加特殊标记
                        window_input_ids = [src_tokenizer.cls_token_id] + input_ids[start_idx:end_idx] + [src_tokenizer.sep_token_id]
                        window_attention_mask = [1] * len(window_input_ids)
                        
                        # 转换为张量并移动到设备
                        window_inputs = {
                            'input_ids': torch.tensor([window_input_ids]).to(device),
                            'attention_mask': torch.tensor([window_attention_mask]).to(device)
                        }
                        
                        # 获取窗口嵌入
                        window_outputs = src_model(**window_inputs)
                        window_embedding = window_outputs.last_hidden_state[:, 0, :]  # 使用CLS标记嵌入
                        window_embeddings.append(window_embedding)
                    
                    # 如果有窗口嵌入，则取平均值；否则使用零向量
                    if window_embeddings:
                        cls_embedding = torch.mean(torch.stack(window_embeddings), dim=0)
                    else:
                        cls_embedding = torch.zeros(1, src_model.config.hidden_size).to(device)
                
                src_embedding_list.append(cls_embedding)

            # 将列表中的所有嵌入堆叠成一个张量
            src_embedding = torch.stack(src_embedding_list).squeeze(1)
            
            # 移除调试信息
            # print(f"训练: 批次大小={len(asm_code)}, asm_embedding.shape={asm_embedding.shape}, src_embedding.shape={src_embedding.shape}")
            # print(f"训练: asm_embedding.dtype={asm_embedding.dtype}, src_embedding.dtype={src_embedding.dtype}")
            
            # 计算对比损失
            loss, similarity_matrix = criterion(asm_embedding, src_embedding)

            # 缩放损失值以适应梯度累积
            loss = loss / accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 增加梯度累积计数器
            accumulation_counter += 1
            
            # 当达到累积步数或处理完最后一个批次时更新参数
            if accumulation_counter == accumulation_steps or batch_idx == len(train_dataloader) - 1:
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(src_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(asm_mapper.parameters(), max_norm=1.0)
                
                # 更新参数
                optimizer.step()
                optimizer.zero_grad()
                
                # 重置计数器
                accumulation_counter = 0

            # 记录原始损失（未缩放）
            total_train_loss += loss.item() * accumulation_steps

            # 打印当前批次的相似度统计信息（只在训练时打印）
            if batch_idx % 5 == 0:  # 每5个批次打印一次
                batch_size = similarity_matrix.size(0)
                
                # 计算未加margin的原始相似度，用于显示
                asm_emb_norm = nn.functional.normalize(asm_embedding, p=2, dim=1)  
                src_emb_norm = nn.functional.normalize(src_embedding, p=2, dim=1)
                raw_similarity = torch.mm(asm_emb_norm, src_emb_norm.T)
                
                # 取对角线上的值作为正样本相似度
                positive_similarities = torch.diag(raw_similarity)
                
                # 取非对角线上的值作为负样本相似度
                mask = ~torch.eye(batch_size, dtype=torch.bool, device=raw_similarity.device)
                negative_similarities = raw_similarity[mask].view(batch_size, -1)

                avg_positive_sim = positive_similarities.mean().item()
                avg_negative_sim = negative_similarities.mean().item()

                # 打印统计信息
                print(f'训练批次 {batch_idx}: 正样本对 = {batch_size}, 负样本对 = {batch_size * (batch_size - 1)}')
                print(f'训练批次 {batch_idx}: 平均正样本相似度 = {avg_positive_sim:.4f}, 平均负样本相似度 = {avg_negative_sim:.4f}, 差距 = {avg_positive_sim - avg_negative_sim:.4f}')
            
            # 清理中间变量以释放内存
            del asm_embedding, src_embedding, similarity_matrix
            if batch_idx % 5 == 0:  # 只在打印批次清理额外的内存
                del raw_similarity, positive_similarities, negative_similarities, mask
            del asm_embedding_list, src_embedding_list
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # 计算平均训练损失
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f'Epoch [{epoch+1}/{epochs}], 训练损失: {avg_train_loss:.4f}')
        
        # ================ 验证阶段 ================
        src_model.eval()
        asm_mapper.eval()
        asm_model.eval()
        total_val_loss = 0
        
        # 用于计算MRR的变量
        mrr_sum = 0
        mrr_count = 0
        
        # 用于计算Top-1和Top-3准确率的变量
        top1_correct = 0
        top3_correct = 0
        total_samples = 0
        
        with torch.no_grad():  # 在验证阶段不计算梯度
            for batch_idx, (asm_code, src_code) in enumerate(tqdm(val_dataloader, desc=f'验证 Epoch {epoch+1}/{epochs}')):
                # 清理 CUDA 缓存
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                # 处理汇编代码 - 使用ASM2VEC模型
                asm_embedding_list = []
                asm_seq_list = []  # 存储序列数据
                asm_seq_lens = []  # 存储序列长度
                
                for code in asm_code:
                    try:
                        # 创建Function对象并获取嵌入向量
                        fn = asm2vec.datatype.Function.load(code)
                        fn_tokens = fn.tokens()
                        
                        # 为每个token添加到词汇表
                        token_ids = []
                        for token in fn_tokens:
                            if token not in tokens.name_to_index:
                                tokens.add(token)
                            token_ids.append(tokens.name_to_index[token])
                        
                        # 使用embeddings_r获取token的嵌入，保留序列信息
                        with torch.no_grad():
                            if token_ids:
                                # 确保所有token_ids都在有效范围内
                                max_idx = asm_model.embeddings_r.num_embeddings - 1
                                valid_token_ids = [min(idx, max_idx) for idx in token_ids]
                                
                                try:
                                    # 首先尝试在GPU上处理
                                    token_tensor = torch.tensor(valid_token_ids, device=device)
                                    token_embeddings = asm_model.embeddings_r(token_tensor)
                                    
                                    # 如果序列足够长，使用序列模式
                                    if len(token_embeddings) > 5:  # 最小长度阈值
                                        if len(token_embeddings) > 1000:  # 截断过长序列
                                            token_embeddings = token_embeddings[:1000]
                                        asm_seq_list.append(token_embeddings)
                                        asm_seq_lens.append(len(token_embeddings))
                                    else:
                                        # 对于短序列使用平均池化
                                        embedding = token_embeddings.mean(dim=0)
                                        asm_embedding_list.append(embedding)
                                except RuntimeError as e:
                                    # 如果GPU出错，尝试在CPU上处理
                                    print(f"GPU处理出错，转为CPU: {e}")
                                    cpu_token_tensor = torch.tensor(valid_token_ids)
                                    cpu_embeddings = asm_model.embeddings_r.cpu()(cpu_token_tensor)
                                    embedding = cpu_embeddings.mean(dim=0).to(device)
                                    asm_embedding_list.append(embedding)
                            else:
                                # 如果没有有效token，使用零向量
                                try:
                                    embedding = torch.zeros(asm_model.embeddings_r.embedding_dim, device=device)
                                except RuntimeError:
                                    # 如果GPU创建失败，使用CPU
                                    embedding = torch.zeros(asm_model.embeddings_r.embedding_dim).to(device)
                            asm_embedding_list.append(embedding)
                    except Exception as e:
                        print(f"验证时处理汇编代码出错: {e}")
                        # 出错时使用零向量
                        try:
                            embedding_size = asm_model.embeddings_r.embedding_dim
                            embedding = torch.zeros(embedding_size, device=device)
                        except RuntimeError:
                            # 如果GPU创建失败，使用CPU
                            embedding_size = asm_model.embeddings_r.embedding_dim
                            embedding = torch.zeros(embedding_size).to(device)
                        asm_embedding_list.append(embedding)
                
                # 处理嵌入
                if asm_seq_list and len(asm_seq_list) == len(asm_code):
                    # 如果所有样本都是序列，使用填充批处理
                    max_len = max(len(seq) for seq in asm_seq_list)
                    padded_seqs = []
                    for seq in asm_seq_list:
                        if len(seq) < max_len:
                            padding = torch.zeros(max_len - len(seq), seq.size(1), device=device)
                            padded_seq = torch.cat([seq, padding], dim=0)
                        else:
                            padded_seq = seq
                        padded_seqs.append(padded_seq)
                    asm_embedding_batch = torch.stack(padded_seqs)
                    # 使用映射层处理批次序列
                    asm_embedding = asm_mapper(asm_embedding_batch, asm_seq_lens)
                elif asm_embedding_list:
                    # 否则使用平均嵌入
                    asm_embedding = torch.stack(asm_embedding_list)
                    # 使用映射层将ASM嵌入映射到与源代码嵌入相同的维度
                    asm_embedding = asm_mapper(asm_embedding)
                else:
                    # 如果两者都为空，创建零张量批次
                    asm_embedding = torch.zeros(len(asm_code), asm_model.embeddings_r.embedding_dim, device=device)
                    asm_embedding = asm_mapper(asm_embedding)
                
                # 处理源代码 - 使用CodeBERT模型
                src_embedding_list = []
                for code in src_code:
                    max_length = 512
                    tokens_bert = src_tokenizer(code, return_tensors='pt', truncation=True, padding=False, max_length=max_length)
                    
                    # 如果代码长度小于最大长度，直接处理
                    if tokens_bert['input_ids'].size(1) <= max_length:
                        inputs = {k: v.to(device) for k, v in tokens_bert.items()}
                        outputs = src_model(**inputs)
                        cls_embedding = outputs.last_hidden_state[:, 0, :]
                    else:
                        # 使用滑动窗口处理长代码
                        # print("验证集：代码长度超出设定的max_length，使用滑动窗口处理")
                        window_size = max_length - 50  # 窗口大小（留出一些重叠空间）
                        stride = window_size // 2  # 滑动步长（50%重叠）
                        
                        # 将代码分词
                        all_tokens = src_tokenizer(code, add_special_tokens=False)
                        input_ids = all_tokens['input_ids']
                        
                        # 存储每个窗口的嵌入向量
                        window_embeddings = []
                        
                        # 分窗口处理
                        for start_idx in range(0, len(input_ids), stride):
                            end_idx = min(start_idx + window_size, len(input_ids))
                            if end_idx - start_idx < 100:  # 跳过太小的窗口
                                continue
                                
                            # 为窗口添加特殊标记
                            window_input_ids = [src_tokenizer.cls_token_id] + input_ids[start_idx:end_idx] + [src_tokenizer.sep_token_id]
                            window_attention_mask = [1] * len(window_input_ids)
                            
                            # 转换为张量并移动到设备
                            window_inputs = {
                                'input_ids': torch.tensor([window_input_ids]).to(device),
                                'attention_mask': torch.tensor([window_attention_mask]).to(device)
                            }
                            
                            # 获取窗口嵌入
                            window_outputs = src_model(**window_inputs)
                            window_embedding = window_outputs.last_hidden_state[:, 0, :]  # 使用CLS标记嵌入
                            window_embeddings.append(window_embedding)
                        
                        # 如果有窗口嵌入，则取平均值；否则使用零向量
                        if window_embeddings:
                            cls_embedding = torch.mean(torch.stack(window_embeddings), dim=0)
                        else:
                            cls_embedding = torch.zeros(1, src_model.config.hidden_size).to(device)
                    
                    src_embedding_list.append(cls_embedding)
                
                # 将列表中的所有嵌入堆叠成一个张量
                src_embedding = torch.stack(src_embedding_list).squeeze(1)
                
                # 移除调试信息
                # print(f"验证: 批次大小={len(asm_code)}, asm_embedding.shape={asm_embedding.shape}, src_embedding.shape={src_embedding.shape}")
                # print(f"验证: asm_embedding.dtype={asm_embedding.dtype}, src_embedding.dtype={src_embedding.dtype}")
                
                # 计算验证损失
                loss, similarity_matrix = criterion(asm_embedding, src_embedding)
                total_val_loss += loss.item()
                
                # 计算MRR（平均倒数排名）
                batch_size = similarity_matrix.size(0)
                for i in range(batch_size):
                    # 获取当前汇编代码与所有源代码的相似度
                    similarities = similarity_matrix[i]
                    
                    # 获取降序排序后的索引（相似度从高到低）
                    _, indices = torch.sort(similarities, descending=True)
                    
                    # 找出正确匹配的排名（正确匹配的索引是i）
                    correct_idx = i
                    rank = torch.where(indices == correct_idx)[0].item() + 1  # +1因为排名从1开始
                    
                    # 累加倒数排名
                    mrr_sum += 1.0 / rank
                    mrr_count += 1
                    
                    # 计算Top-1准确率（排名为1表示正确匹配）
                    if rank == 1:
                        top1_correct += 1
                    
                    # 计算Top-3准确率（排名在前3表示命中）
                    if rank <= 3:
                        top3_correct += 1
                    
                    total_samples += 1
                
                # 清理内存
                del asm_embedding, src_embedding, asm_embedding_list, src_embedding_list
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        # 计算平均验证损失
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        # 计算MRR和Top-N准确率
        epoch_mrr = mrr_sum / mrr_count if mrr_count > 0 else 0
        top1_accuracy = top1_correct / total_samples if total_samples > 0 else 0
        top3_accuracy = top3_correct / total_samples if total_samples > 0 else 0
        
        print(f'Epoch [{epoch+1}/{epochs}], 验证损失: {avg_val_loss:.4f}, MRR: {epoch_mrr:.4f}, '
              f'Top-1准确率: {top1_accuracy:.4f}, Top-3准确率: {top3_accuracy:.4f}')
        
        # 更新MRR历史
        mrr_history.append(epoch_mrr)
        global_vars['mrr_history'] = mrr_history
        
        # 更新Top-1和Top-3准确率历史
        top1_history.append(top1_accuracy)
        top3_history.append(top3_accuracy)
        global_vars['top1_history'] = top1_history
        global_vars['top3_history'] = top3_history
        
        # 更新学习率
        scheduler.step()
        
        # 每个epoch结束后保存模型
        save_model(output_path, asm_model, src_model, asm_mapper, epoch, train_losses, val_losses, mrr_history, 
                  top1_history, top3_history)
        
        # 每个epoch结束后绘制所有指标曲线
        save_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        plot_all_metrics(
            train_losses, 
            val_losses, 
            mrr_history, 
            top1_history, 
            top3_history, 
            save_dir
        )
        
        # 每个 epoch 结束后清理 CUDA 缓存
        if device == 'cuda':
            torch.cuda.empty_cache()

    # 训练完成后保存最终模型
    save_model(output_path, asm_model, src_model, asm_mapper, epochs-1, train_losses, val_losses, mrr_history, 
              top1_history, top3_history)
    print(f"训练完成，最终模型已保存到 {output_path}")
    
    # 绘制最终的所有指标曲线
    save_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
    plot_all_metrics(
        train_losses, 
        val_losses, 
        mrr_history, 
        top1_history, 
        top3_history, 
        save_dir
    )

    torch.cuda.empty_cache()

if __name__ == '__main__':
    train_model() 