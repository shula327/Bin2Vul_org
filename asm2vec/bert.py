import torch
from transformers import RobertaTokenizer, RobertaModel

# 载入 CodeBERT 模型和对应的 tokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base')

def code_to_embedding(code: str):
    # 对源码进行编码
    inputs = tokenizer(code, return_tensors='pt', truncation=True, padding=True)
    
    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取代码的嵌入向量（一般使用 [CLS] token 对应的嵌入向量）
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze()
    
    return embeddings

# 测试源码
code_example = '''
def hello_world():
    print("Hello, World!")
'''

# 获取源码的嵌入向量
embedding = code_to_embedding(code_example)

# 将 PyTorch tensor 转换为 NumPy 数组并输出
embedding_numpy = embedding.numpy()

# 输出嵌入向量的数值
print("Embedding vector:", embedding_numpy)
print("Embedding shape:", embedding_numpy.shape)
