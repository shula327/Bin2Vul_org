import torch
import torch.nn as nn

# 定义损失函数和激活函数
bce, sigmoid, softmax = nn.BCELoss(), nn.Sigmoid(), nn.Softmax(dim=1)

class ASM2VEC(nn.Module):
    def __init__(self, vocab_size, function_size, embedding_size):
        super(ASM2VEC, self).__init__()
        #词汇嵌入，初始化为零向量(vocab_size=50200, embedding_size=128)(50200,128)
        self.embeddings = nn.Embedding(vocab_size, embedding_size, _weight=torch.zeros(vocab_size, embedding_size))  # 词汇嵌入
        #函数嵌入(function_size=1000, 2 * embedding_size=256)，随机初始化，用于全局表示(1000,256)
        self.embeddings_f = nn.Embedding(function_size, 2 * embedding_size, _weight=(torch.rand(function_size, 2 * embedding_size) - 0.5) / embedding_size / 2)  # 函数嵌入
        #反向词汇嵌入(vocab_size=50200, 2 * embedding_size=256)(50200,256)
        self.embeddings_r = nn.Embedding(vocab_size, 2 * embedding_size, _weight=(torch.rand(vocab_size, 2 * embedding_size) - 0.5) / embedding_size / 2)  # 反向词嵌入

    def v(self, inp):
        batch_size, seq_len = inp.size(0), inp.size(1)
        window_size = 6  # 窗口大小
        embedding_dim = self.embeddings.embedding_dim
        device = inp.device

    # 初始化一个列表来存储每个窗口的嵌入向量
        window_embeddings = []

    # 滑动窗口的起始位置
        for i in range(0, seq_len - 1, window_size):  # 滑动步长为 window_size
        # 获取当前窗口的词汇
            window = inp[:, i+1:i+1+window_size]  # 跳过函数标识符 inp[:, 0]
        
        # 如果窗口大小不足，跳过
            if window.size(1) < 3:
                continue

        # 计算当前窗口的嵌入向量
            e = self.embeddings(window)  # 词汇嵌入
            v_f = self.embeddings_f(inp[:, 0])  # 函数嵌入

        # 检查 e 的序列长度是否足够
            if e.size(1) >= 6:
                v_prev = torch.cat([e[:, 0], (e[:, 1] + e[:, 2]) / 2], dim=1)
                v_next = torch.cat([e[:, 3], (e[:, 4] + e[:, 5]) / 2], dim=1)
            elif e.size(1) >= 4:
            # 如果序列长度不足6但至少有4个词汇，只计算 v_prev
                v_prev = torch.cat([e[:, 0], (e[:, 1] + e[:, 2]) / 2], dim=1)
                v_next = torch.zeros_like(v_prev)  # 用零向量填充 v_next
            else:
            # 如果序列长度不足4，跳过这个窗口
                continue

            v_window = ((v_f + v_prev + v_next) / 3).unsqueeze(2)
        
        # 将当前窗口的嵌入向量添加到列表中
            window_embeddings.append(v_window)

    # 如果没有有效的窗口，返回零向量
        if len(window_embeddings) == 0:
            return torch.zeros(batch_size, 2 * embedding_dim, 1).to(device)

    # 对所有窗口的嵌入向量取均值
        v = torch.mean(torch.stack(window_embeddings, dim=0), dim=0)
        return v

  
    def update(self, function_size_new, vocab_size_new):
        # 更新嵌入层的大小
        device = self.embeddings.weight.device  # 获取当前设备
        vocab_size, function_size, embedding_size = self.embeddings.num_embeddings, self.embeddings_f.num_embeddings, self.embeddings.embedding_dim  # 获取当前的词汇大小、函数大小和嵌入维度
        if vocab_size_new != vocab_size:  # 如果新的词汇大小与当前不同
            # 扩展词汇嵌入
            weight = torch.cat([self.embeddings.weight, torch.zeros(vocab_size_new - vocab_size, embedding_size).to(device)])  # 扩展权重
            self.embeddings = nn.Embedding(vocab_size_new, embedding_size, _weight=weight)  # 更新词汇嵌入层
            weight_r = torch.cat([self.embeddings_r.weight, ((torch.rand(vocab_size_new - vocab_size, 2 * embedding_size) - 0.5) / embedding_size / 2).to(device)])  # 扩展反向词嵌入权重
            self.embeddings_r = nn.Embedding(vocab_size_new, 2 * embedding_size, _weight=weight_r)  # 更新反向词嵌入层
        # 更新函数嵌入
        self.embeddings_f = nn.Embedding(function_size_new, 2 * embedding_size, _weight=((torch.rand(function_size_new, 2 * embedding_size) - 0.5) / embedding_size / 2).to(device))

    def forward(self, inp, pos, neg):
        # 前向传播
        device, batch_size = inp.device, inp.shape[0]  # 获取设备和批次大小
        v = self.v(inp)  # 计算嵌入向量
        # 负采样损失
        pred = torch.bmm(self.embeddings_r(torch.cat([pos, neg], dim=1)), v).squeeze()  # 计算预测值
        label = torch.cat([torch.ones(batch_size, 3), torch.zeros(batch_size, neg.shape[1])], dim=1).to(device)  # 创建标签
        return bce(sigmoid(pred), label)  # 返回损失值

    def predict(self, inp, pos):
        # 进行预测
        device, batch_size = inp.device, inp.shape[0]  # 获取设备和批次大小
        v = self.v(inp)  # 计算嵌入向量
        # 计算每个词的概率
        probs = torch.bmm(self.embeddings_r(torch.arange(self.embeddings_r.num_embeddings).repeat(batch_size, 1).to(device)), v).squeeze(dim=2)  # 计算概率
        return softmax(probs)  # 返回经过softmax处理的概率
