import torch
import torch.nn as nn

# 定义损失函数和激活函数
bce, sigmoid, softmax = nn.BCELoss(), nn.Sigmoid(), nn.Softmax(dim=1)

class ASM2VEC(nn.Module):
    def __init__(self, vocab_size, function_size, embedding_size):
        super(ASM2VEC, self).__init__()
        # 初始化嵌入层
        #用于将输入的词汇索引映射到对应的嵌入向量，初始化为全零
        self.embeddings = nn.Embedding(vocab_size, embedding_size, _weight=torch.zeros(vocab_size, embedding_size))
        #输入的函数索引映射到对应的嵌入向量，初始化为random
        self.embeddings_f = nn.Embedding(function_size, 2 * embedding_size, _weight=(torch.rand(function_size, 2 * embedding_size) - 0.5) / embedding_size / 2)
        #输入的反向词汇索引映射到对应的嵌入向量，初始化为随机值矩阵。
        self.embeddings_r = nn.Embedding(vocab_size, 2 * embedding_size, _weight=(torch.rand(vocab_size, 2 * embedding_size) - 0.5) / embedding_size / 2)

    def update(self, function_size_new, vocab_size_new):
        # 更新嵌入层的大小
        device = self.embeddings.weight.device
        vocab_size, function_size, embedding_size = self.embeddings.num_embeddings, self.embeddings_f.num_embeddings, self.embeddings.embedding_dim
        if vocab_size_new != vocab_size:
            # 更新词汇表嵌入层
            weight = torch.cat([self.embeddings.weight, torch.zeros(vocab_size_new - vocab_size, embedding_size).to(device)])
            self.embeddings = nn.Embedding(vocab_size_new, embedding_size, _weight=weight)
            # 更新反向嵌入层，将当前反向嵌入层的权重与新的随机权重拼接在一起，以适应新的词汇表大小。
            #新的随机权重在 [-0.5, 0.5] 范围内，并且被缩放以适应嵌入向量的维度。
            weight_r = torch.cat([self.embeddings_r.weight, ((torch.rand(vocab_size_new - vocab_size, 2 * embedding_size) - 0.5) / embedding_size / 2).to(device)])
            #新建一个反向嵌入层
            self.embeddings_r = nn.Embedding(vocab_size_new, 2 * embedding_size, _weight=weight_r)
        # 更新函数嵌入层
        self.embeddings_f = nn.Embedding(function_size_new, 2 * embedding_size, _weight=((torch.rand(function_size_new, 2 * embedding_size) - 0.5) / embedding_size / 2).to(device))

    def v(self, inp):
        # 计算输入的嵌入向量
        e = self.embeddings(inp[:, 1:])
        max_index = self.embeddings.num_embeddings - 1
        if e.size(1) < 3:
            # 如果长度小于3，返回一个默认值或处理逻辑
            return torch.zeros(inp.size(0), 256).to(inp.device)  # 返回零向量或其他处理

        #inp[:;1:;]:[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        #词汇向量，example：[2, 5, 128]
        e = self.embeddings(inp[:, 1:])
       #函数本身的嵌入张量,[2, 256]
        v_f = self.embeddings_f(inp[:, 0])
       #前一个词汇的嵌入张量,[2，128]，[2,128]-[2,256]
        v_prev = torch.cat([e[:, 0], (e[:, 1] + e[:, 2]) / 2], dim=1)
        #后一个词汇的嵌入张量
        v_next = torch.cat([e[:, 3], (e[:, 4] + e[:, 5]) / 2], dim=1)
        v = ((v_f + v_prev + v_next) / 3).unsqueeze(2)
        return v

    def forward(self, inp, pos, neg):
        # 前向传播计算损失
        device, batch_size = inp.device, inp.shape[0]
        v = self.v(inp)
        # 负采样损失
        pred = torch.bmm(self.embeddings_r(torch.cat([pos, neg], dim=1)), v).squeeze()
        label = torch.cat([torch.ones(batch_size, 3), torch.zeros(batch_size, neg.shape[1])], dim=1).to(device)
        return bce(sigmoid(pred), label)

    def predict(self, inp, pos):
        # 预测函数
        device, batch_size = inp.device, inp.shape[0]
        v = self.v(inp)
        probs = torch.bmm(self.embeddings_r(torch.arange(self.embeddings_r.num_embeddings).repeat(batch_size, 1).to(device)), v).squeeze(dim=2)
        return softmax(probs)
