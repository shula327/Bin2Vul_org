from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import CustomUserCreationForm, CustomAuthenticationForm, BinaryCompareForm, VulnerabilityAnalysisForm, CustomPasswordChangeForm
from .models import AnalysisHistory
import os
import subprocess
from django.conf import settings
import uuid
import time
import json
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from pathlib import Path
import re
import r2pipe
import tempfile
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import hashlib

def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, '注册成功！')
            return redirect('dashboard')
    else:
        form = CustomUserCreationForm()
    return render(request, 'core/register.html', {'form': form})

def login_view(request):
    # 获取用户名（POST优先，GET为空）
    username = request.POST.get('username', '').strip() if request.method == 'POST' else ''
    # 用session记录每个用户名的失败次数和冻结时间
    fail_key = f'login_fail_{username}'
    freeze_key = f'login_freeze_{username}'
    fail_count = request.session.get(fail_key, 0)
    freeze_until = request.session.get(freeze_key, 0)
    now = int(time.time())
    # 检查是否被冻结
    if freeze_until and now < freeze_until:
        left = freeze_until - now
        messages.error(request, f'该账号因多次登录失败已被冻结，请{left}秒后再试。')
        form = CustomAuthenticationForm(request, data=request.POST or None)
        return render(request, 'core/login.html', {'form': form})
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            # 登录成功，清除失败计数和冻结
            if fail_key in request.session:
                del request.session[fail_key]
            if freeze_key in request.session:
                del request.session[freeze_key]
            messages.success(request, '登录成功！')
            return redirect('dashboard')
        else:
            # 登录失败，增加计数
            fail_count += 1
            request.session[fail_key] = fail_count
            if fail_count >= 5:
                # 冻结1分钟
                request.session[freeze_key] = now + 60
                messages.error(request, '登录失败次数过多，账号已冻结1分钟，请稍后再试。')
            else:
                left = 5 - fail_count
                messages.error(request, f'用户名、密码或验证码错误，您还可以尝试{left}次。')
    else:
        form = CustomAuthenticationForm()
    return render(request, 'core/login.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.success(request, '已成功退出！')
    return redirect('login')

@login_required
def dashboard_view(request):
    history = AnalysisHistory.objects.filter(user=request.user).order_by('-created_at')[:5]
    return render(request, 'core/dashboard.html', {'history': history})

@login_required
def binary_compare_view(request):
    if request.method == 'POST':
        form = BinaryCompareForm(request.POST, request.FILES)
        if form.is_valid():
            # 创建临时目录
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            asm_output_dir = os.path.join(temp_dir, 'asm_output')
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(asm_output_dir, exist_ok=True)
            
            # 获取脚本所在的根目录（asm2vec-pytorch）
            scripts_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # 使用安全的文件名
            file1_bytes = b''.join(chunk for chunk in request.FILES['file1'].chunks())
            file2_bytes = b''.join(chunk for chunk in request.FILES['file2'].chunks())
            hash1 = hashlib.sha256(file1_bytes).hexdigest()
            hash2 = hashlib.sha256(file2_bytes).hexdigest()
            if hash1 == hash2:
                file1_name = file2_name = 'samefile.elf'
            else:
                file1_name = f"{uuid.uuid4().hex}_1.elf"
                file2_name = f"{uuid.uuid4().hex}_2.elf"
            file1_path = os.path.join(temp_dir, file1_name)
            file2_path = os.path.join(temp_dir, file2_name)
            
            try:
                # 保存文件
                with open(file1_path, 'wb') as f:
                    f.write(file1_bytes)
                with open(file2_path, 'wb') as f:
                    f.write(file2_bytes)

                # 验证文件格式
                def validEXE(filename):
                    """检查文件是否为有效的可执行文件（支持ELF和PE格式）"""
                    # 魔数定义
                    ELF_MAGIC = bytes.fromhex('7f454c46')  # ELF格式
                    PE_MAGIC = bytes.fromhex('4d5a')       # MZ (DOS/PE格式)
                    
                    try:
                        with open(filename, 'rb') as f:
                            # 读取文件头
                            header = f.read(4)
                            
                            # 检查ELF格式
                            if header.startswith(ELF_MAGIC):
                                return True
                                
                            # 检查PE格式
                            if header.startswith(PE_MAGIC):
                                return True
                                
                        return False
                    except Exception as e:
                        print(f"[Warning] 检查文件格式时出错 {filename}: {str(e)}")
                        return False

                # 验证两个文件
                if not validEXE(file1_path):
                    raise ValueError(f"第一个文件不是有效的可执行文件")
                if not validEXE(file2_path):
                    raise ValueError(f"第二个文件不是有效的可执行文件")

                # 检查compare408_debug.py是否存在
                compare_script = os.path.join(scripts_root, 'compare408_debug.py')
                if not os.path.exists(compare_script):
                    raise FileNotFoundError(f"比较脚本不存在: {compare_script}")

                # 检查模型文件
                model_path = os.path.join(scripts_root, 'model407.pt')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"模型文件不存在: {model_path}")

                # 构建命令
                cmd = [
                    'python',
                    compare_script,
                    '-i1', file1_path,
                    '-i2', file2_path,
                    '-o', asm_output_dir,
                    '-m', model_path,
                    '-e', '5',
                    '-c', 'auto',
                    '-lr', '0.001'
                ]

                print(f"执行命令: {' '.join(cmd)}")  # 打印完整命令
                
                # 检查bin2asm318.py是否存在
                bin2asm_script = os.path.join(scripts_root, 'bin2asm318.py')
                if not os.path.exists(bin2asm_script):
                    raise FileNotFoundError(f"二进制转汇编脚本不存在: {bin2asm_script}")
                
                # 检查test408_debug.py是否存在
                test_script = os.path.join(scripts_root, 'test408_debug.py')
                if not os.path.exists(test_script):
                    raise FileNotFoundError(f"测试脚本不存在: {test_script}")
                
                # 确保输出目录存在
                os.makedirs(asm_output_dir, exist_ok=True)
                
                # 检查输出目录权限
                try:
                    test_file = os.path.join(asm_output_dir, 'test_write.tmp')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                except Exception as e:
                    raise PermissionError(f"输出目录没有写入权限: {asm_output_dir}, 错误: {str(e)}")

                result = subprocess.run(cmd, 
                                    capture_output=True, 
                                    text=True, 
                                    check=False,
                                    cwd=scripts_root)

                if result.returncode != 0:
                    error_msg = f"命令执行失败:\n"
                    error_msg += f"返回码: {result.returncode}\n"
                    error_msg += f"标准输出: {result.stdout}\n"
                    error_msg += f"错误输出: {result.stderr}\n"
                    error_msg += f"工作目录: {scripts_root}\n"
                    error_msg += f"相关文件检查:\n"
                    error_msg += f"- compare408_debug.py 存在: {os.path.exists(compare_script)}\n"
                    error_msg += f"- bin2asm318.py 存在: {os.path.exists(bin2asm_script)}\n"
                    error_msg += f"- test408_debug.py 存在: {os.path.exists(test_script)}\n"
                    error_msg += f"- model407.pt 存在: {os.path.exists(model_path)}\n"
                    error_msg += f"- 输出目录存在: {os.path.exists(asm_output_dir)}\n"
                    print(error_msg)  # 打印到服务器日志
                    
                    # 尝试直接运行bin2asm318.py来获取更多信息
                    try:
                        bin2asm_result = subprocess.run(
                            ['python', bin2asm_script, '-i', file1_path, '-o', asm_output_dir, '-l', '0'],
                            capture_output=True,
                            text=True,
                            check=False,
                            cwd=scripts_root
                        )
                        error_msg += f"\n尝试直接运行bin2asm318.py的结果:\n"
                        error_msg += f"返回码: {bin2asm_result.returncode}\n"
                        error_msg += f"输出: {bin2asm_result.stdout}\n"
                        error_msg += f"错误: {bin2asm_result.stderr}\n"
                        print(error_msg)
                    except Exception as e:
                        error_msg += f"\n运行bin2asm318.py时出错: {str(e)}\n"
                        print(error_msg)
                    
                    raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

                # 解析输出获取相似度和函数对信息
                lines = result.stdout.strip().split('\n')
                similarity_score = None
                top_pairs = []
                parsing_pairs = False

                for line in lines:
                    if 'cosine similarity :' in line:
                        similarity_score = float(line.split(':')[1].strip())
                    elif line.startswith('Top 5 most similar function pairs:'):
                        parsing_pairs = True
                    elif parsing_pairs and ':' in line:
                        # 支持三种格式：func1 - func2: score、func1 (self): score、func1: score（同源时可能只有一个函数名）
                        if ' - ' in line:
                            funcs, score = line.rsplit(':', 1)
                            func1, func2 = funcs.split(' - ')
                            score = float(score.strip())
                            top_pairs.append({
                                'func1': func1.strip(),
                                'func2': func2.strip(),
                                'score': score,
                                'percentage': (score + 1) * 50
                            })
                        elif '(self):' in line:
                            func1_part, score = line.rsplit(':', 1)
                            func1 = func1_part.replace('(self)', '').strip()
                            score = float(score.strip())
                            top_pairs.append({
                                'func1': func1,
                                'func2': func1,
                                'score': score,
                                'percentage': (score + 1) * 50
                            })
                        else:
                            # 形如 func1: score（同源时可能只输出函数名:分数）
                            func1, score = line.rsplit(':', 1)
                            func1 = func1.strip()
                            score = float(score.strip())
                            top_pairs.append({
                                'func1': func1,
                                'func2': func1,
                                'score': score,
                                'percentage': (score + 1) * 50
                            })

                if similarity_score is not None:
                    # 保存分析历史
                    analysis = AnalysisHistory.objects.create(
                        user=request.user,
                        analysis_type='binary_compare',
                        file1=request.FILES['file1'],
                        file2=request.FILES['file2'],
                        similarity_score=similarity_score,
                        top_results=json.dumps(top_pairs)  # 保存函数对信息
                    )

                    # 计算总体相似度的百分比
                    similarity_percentage = (similarity_score + 1) * 50

                    messages.success(request, f'分析完成！相似度得分：{similarity_percentage:.2f}%')
                    return render(request, 'core/binary_compare.html', {
                        'form': form,
                        'similarity_score': similarity_score,
                        'similarity_percentage': similarity_percentage,
                        'top_pairs': top_pairs  # 传递函数对信息到模板
                    })
                else:
                    messages.error(request, '无法解析相似度得分')

            except Exception as e:
                messages.error(request, f'分析过程中出现错误：{str(e)}')
            finally:
                # 清理临时文件
                try:
                    if os.path.exists(file1_path):
                        os.remove(file1_path)
                    if os.path.exists(file2_path):
                        os.remove(file2_path)
                    if os.path.exists(asm_output_dir):
                        for file in os.listdir(asm_output_dir):
                            os.remove(os.path.join(asm_output_dir, file))
                        os.rmdir(asm_output_dir)
                except Exception as e:
                    print(f"清理临时文件时出错: {e}")
    else:
        form = BinaryCompareForm()
    
    return render(request, 'core/binary_compare.html', {'form': form})

@login_required
def vulnerability_analysis_view(request):
    if request.method == 'POST':
        form = VulnerabilityAnalysisForm(request.POST, request.FILES)
        if form.is_valid():
            # 创建临时目录
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # 获取脚本所在的根目录
            scripts_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # 使用安全的文件名
            file_name = f"{uuid.uuid4().hex}_binary"
            file_path = os.path.join(temp_dir, file_name)
            
            try:
                # 保存上传的文件
                with open(file_path, 'wb+') as f:
                    for chunk in request.FILES['file'].chunks():
                        f.write(chunk)
                
                # 漏洞源码目录
                vuln_src_dir = os.path.join(scripts_root, 'data/src1/')
                
                # 加载模型
                print("正在加载模型...")
                asm_model_path = os.path.join(scripts_root, "model420.pt")  # ASM2VEC模型
                model_path = os.path.join(scripts_root, "model_asm2vec5012.pt")  # 映射模型
                
                # 设置设备
                device = 'cpu'
                # device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"使用设备: {device}")
                
                # 加载ASM2VEC预训练模型
                print("加载ASM2VEC预训练模型...")
                import asm2vec
                asm_model, tokens = asm2vec.utils.load_model(asm_model_path, device=device)
                
                # 初始化源代码模型
                print("初始化CodeBERT模型...")
                from transformers import RobertaTokenizer, RobertaModel
                src_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
                src_model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
                
                # 加载训练好的模型
                print("加载训练好的模型状态...")
                checkpoint = torch.load(model_path, map_location=device)
                
                # 加载源代码模型状态
                if 'src_model_state_dict' in checkpoint:
                    src_model.load_state_dict(checkpoint['src_model_state_dict'])
                    print("已加载源代码模型状态")
                
                # 创建ASM嵌入映射层
                asm_embedding_dim = asm_model.embeddings_r.embedding_dim  # 通常是400
                src_embedding_dim = src_model.config.hidden_size  # CodeBERT是768
                
                # 定义ASM嵌入映射层
                from pathlib import Path
                import math
                
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
                        # x: [seq_len, batch_size, d_model]
                        x = x + self.pe[:, :x.size(0)].transpose(0, 1)
                        return self.dropout(x)
                
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
                            dropout=dropout
                        )
                        
                    def forward(self, x, seq_lens=None):
                        # 检查是否是批处理的序列数据还是单个向量
                        is_single_vector = x.dim() == 1 or (x.dim() == 2 and not seq_lens)
                        
                        if is_single_vector:
                            # 如果是单个向量（平均池化后的结果），直接映射
                            return self.linear(x)
                        
                        # 确保输入是3D: [seq_len, batch_size, feature_dim] - 早期PyTorch的Transformer要求
                        if x.dim() == 2:  # [seq_len, feature_dim]
                            x = x.unsqueeze(1)  # 添加批次维度 [seq_len, 1, feature_dim]
                        elif x.dim() == 3 and x.size(0) != x.size(1):  # [batch_size, seq_len, feature_dim]
                            x = x.transpose(0, 1)  # 变成 [seq_len, batch_size, feature_dim]
                        
                        # 应用位置编码和Transformer
                        x = self.position_encoder(x)
                        x = self.transformer_encoder(x)
                        
                        # 转换回 [batch_size, seq_len, feature_dim] 以便池化
                        if x.size(1) != 1:  # 如果批次维度不是1
                            x = x.transpose(0, 1)
                        
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
                
                asm_mapper = ASMEmbeddingMapper(asm_embedding_dim, src_embedding_dim).to(device)
                
                # 加载ASM映射器状态
                if 'asm_mapper_state_dict' in checkpoint:
                    asm_mapper.load_state_dict(checkpoint['asm_mapper_state_dict'])
                    print("已加载ASM映射器状态")
                
                # 设置模型为评估模式
                asm_model.eval()
                src_model.eval()
                asm_mapper.eval()
                
                # 反汇编二进制文件
                print(f"反汇编二进制文件: {file_path}")
                import re
                import hashlib
                import r2pipe
                
                def sha3(data):
                    return hashlib.sha3_256(data.encode()).hexdigest()
                
                def validEXE(filename):
                    """检查文件是否为有效的可执行文件（支持ELF和PE格式）"""
                    # 魔数定义
                    ELF_MAGIC = bytes.fromhex('7f454c46')  # ELF格式
                    PE_MAGIC = bytes.fromhex('4d5a')       # MZ (DOS/PE格式)
                    
                    try:
                        with open(filename, 'rb') as f:
                            # 读取文件头
                            header = f.read(4)
                            
                            # 检查ELF格式
                            if header.startswith(ELF_MAGIC):
                                return True
                                
                            # 检查PE格式
                            if header.startswith(PE_MAGIC):
                                return True
                                
                        return False
                    except Exception as e:
                        print(f"[Warning] 检查文件格式时出错 {filename}: {str(e)}")
                        return False
                
                def normalize(opcode):
                    opcode = opcode.replace(' - ', ' + ')
                    opcode = re.sub(r'0x[0-9a-f]+', 'CONST', opcode)
                    opcode = re.sub(r'\*[0-9]', '*CONST', opcode)
                    opcode = re.sub(r' [0-9]', ' CONST', opcode)
                    return opcode
                
                def fn2asm(pdf, minlen=10):
                    if pdf is None:
                        return ""
                    if len(pdf['ops']) < minlen:
                        return ""
                    if 'invalid' in [op['type'] for op in pdf['ops']]:
                        return ""

                    ops = pdf['ops']
                    labels, scope = {}, [op['offset'] for op in ops]
                    assert(None not in scope)
                    for i, op in enumerate(ops):
                        if op.get('jump') in scope:
                            labels.setdefault(op.get('jump'), i)
                    
                    output = ''
                    for op in ops:
                        if labels.get(op.get('offset')) is not None:
                            output += f'LABEL{labels[op["offset"]]}:\n'
                        if labels.get(op.get('jump')) is not None:
                            output += f' {op["type"]} LABEL{labels[op["jump"]]}\n'
                        else:
                            output += f' {normalize(op["opcode"])}\n'

                    return output
                
                def disassemble_binary(binary_path, minlen=10):
                    """使用与bin2asms.py相同方式反汇编二进制文件，但分别返回每个函数的信息"""
                    try:
                        # 检查是否为二进制文件
                        if not validEXE(binary_path):
                            print(f"错误: {binary_path} 不是有效的可执行文件")
                            return None
                        
                        # 创建临时目录用于存储反汇编结果
                        temp_dir = tempfile.mkdtemp()
                        base_name = os.path.basename(binary_path)
                        
                        # 使用r2pipe反汇编
                        print(f"反汇编二进制文件: {binary_path}")
                        r = r2pipe.open(binary_path)
                        r.cmd('aaaa')  # 使用深度分析
                        
                        functions = []  # 存储所有函数的信息
                        count = 0
                        
                        # 处理每个函数
                        for fn in r.cmdj('aflj'):
                            r.cmd(f's {fn["offset"]}')
                            asm = fn2asm(r.cmdj('pdfj'), minlen)
                            if asm:
                                # 保存函数信息
                                temp_asm_path = os.path.join(temp_dir, f"{base_name}_{fn['name']}.s")
                                with open(temp_asm_path, 'w', encoding='utf-8') as f:
                                    f.write(asm)
                                
                                functions.append({
                                    'name': fn['name'],
                                    'offset': fn['offset'],
                                    'size': fn['size'],
                                    'path': temp_asm_path,
                                    'asm_code': asm
                                })
                                count += 1
                        
                        print(f"已提取 {count} 个函数")
                        if functions:
                            return temp_dir, functions
                        else:
                            print(f"警告: 未从 {binary_path} 提取到有效的汇编函数")
                            return None
                            
                    except Exception as e:
                        print(f"反汇编过程出错: {e}")
                        return None
                
                # 反汇编二进制文件，获取所有函数
                disasm_result = disassemble_binary(file_path)
                if disasm_result is None:
                    raise ValueError("无法反汇编二进制文件")
                
                temp_dir_asm, asm_functions = disasm_result
                print(f"反汇编成功，提取了 {len(asm_functions)} 个函数")
                
                # 定义获取嵌入向量的函数
                def get_asm_embedding(asm_code, asm_model, tokens, asm_mapper, device, use_position=False):
                    """处理汇编代码并返回嵌入向量"""
                    try:
                        # 创建Function对象并获取嵌入向量
                        fn = asm2vec.datatype.Function.load(asm_code)
                        fn_tokens = fn.tokens()
                        
                        # 为每个token添加到词汇表
                        token_ids = []
                        for token in fn_tokens:
                            if token not in tokens.name_to_index:
                                tokens.add(token)
                            token_ids.append(tokens.name_to_index[token])
                        
                        # 使用embeddings_r获取token的嵌入
                        with torch.no_grad():
                            if token_ids:
                                # 确保所有token_ids都在有效范围内
                                max_idx = asm_model.embeddings_r.num_embeddings - 1
                                valid_token_ids = [min(idx, max_idx) for idx in token_ids]
                                
                                try:
                                    # 首先尝试在GPU上处理
                                    token_tensor = torch.tensor(valid_token_ids, device=device)
                                    token_embeddings = asm_model.embeddings_r(token_tensor)
                                    
                                    if use_position and len(token_embeddings) > 5:
                                        # 如果使用位置编码并且序列足够长
                                        if len(token_embeddings) > 1000:  # 截断过长序列
                                            token_embeddings = token_embeddings[:1000]
                                        # 使用映射器处理序列
                                        asm_embedding = asm_mapper(token_embeddings, [len(token_embeddings)])
                                        asm_embedding = asm_embedding.squeeze(0)  # 移除批次维度
                                    else:
                                        # 使用平均池化
                                        embedding = token_embeddings.mean(dim=0)
                                        asm_embedding = asm_mapper(embedding)
                                except RuntimeError as e:
                                    print(f"GPU处理出错，转为CPU: {e}")
                                    cpu_token_tensor = torch.tensor(valid_token_ids)
                                    cpu_embeddings = asm_model.embeddings_r.cpu()(cpu_token_tensor)
                                    embedding = cpu_embeddings.mean(dim=0).to(device)
                                    asm_embedding = asm_mapper(embedding)
                            else:
                                # 如果没有有效token，使用零向量
                                embedding = torch.zeros(asm_model.embeddings_r.embedding_dim, device=device)
                                asm_embedding = asm_mapper(embedding)
                        
                        return asm_embedding
                    except Exception as e:
                        print(f"处理汇编代码时出错: {e}")
                        # 使用零向量作为嵌入
                        return torch.zeros(asm_mapper.linear.out_features, device=device)
                
                def get_src_embedding(src_code, src_tokenizer, src_model, device):
                    """处理源代码并返回嵌入向量"""
                    try:
                        max_length = 512
                        tokens_bert = src_tokenizer(src_code, return_tensors='pt', truncation=True, padding=False, max_length=max_length)
                        
                        # 如果代码长度小于最大长度，直接处理
                        if tokens_bert['input_ids'].size(1) <= max_length:
                            with torch.no_grad():
                                inputs = {k: v.to(device) for k, v in tokens_bert.items()}
                                outputs = src_model(**inputs)
                                src_embedding = outputs.last_hidden_state[:, 0, :]
                        else:
                            print("源代码长度超出设定的max_length，使用滑动窗口处理")
                            window_size = max_length - 50  # 窗口大小（留出一些重叠空间）
                            stride = window_size // 2  # 滑动步长（50%重叠）
                            
                            # 将代码分词
                            all_tokens = src_tokenizer(src_code, add_special_tokens=False)
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
                                with torch.no_grad():
                                    window_outputs = src_model(**window_inputs)
                                    window_embedding = window_outputs.last_hidden_state[:, 0, :]  # 使用CLS标记嵌入
                                window_embeddings.append(window_embedding)
                            
                            # 如果有窗口嵌入，则取平均值；否则使用零向量
                            if window_embeddings:
                                src_embedding = torch.mean(torch.stack(window_embeddings), dim=0)
                            else:
                                src_embedding = torch.zeros(1, src_model.config.hidden_size).to(device)
                        
                        # 确保嵌入维度正确
                        if src_embedding.dim() > 1 and src_embedding.size(0) == 1:
                            src_embedding = src_embedding.squeeze(0)
                            
                        return src_embedding
                    except Exception as e:
                        print(f"处理源代码时出错: {e}")
                        # 使用零向量作为嵌入
                        return torch.zeros(src_model.config.hidden_size, device=device)
                
                def calculate_similarity(vec1, vec2):
                    """计算两个向量的余弦相似度"""
                    with torch.no_grad():
                        # 确保输入是一维向量
                        if vec1.dim() > 1:
                            vec1 = vec1.mean(dim=0)  # 对序列进行平均池化
                        
                        if vec2.dim() > 1:
                            vec2 = vec2.mean(dim=0)  # 对序列进行平均池化
                            
                        # 归一化向量
                        vec1_norm = nn.functional.normalize(vec1, p=2, dim=0)
                        vec2_norm = nn.functional.normalize(vec2, p=2, dim=0)
                        
                        # 计算余弦相似度
                        return torch.dot(vec1_norm, vec2_norm).item()
                
                # 查找所有漏洞源码文件
                vuln_files = []
                for ext in ['.c', '.cpp', '.h', '.cc']:
                    vuln_files.extend(Path(vuln_src_dir).glob(f'*{ext}'))
                
                if not vuln_files:
                    raise FileNotFoundError(f"在 {vuln_src_dir} 中未找到任何源码文件")
                
                print(f"找到 {len(vuln_files)} 个漏洞源码文件")
                
                # 预先计算所有源代码嵌入向量
                print("计算所有源代码文件的嵌入向量...")
                src_embeddings = {}
                
                from tqdm import tqdm
                for src_path in tqdm(vuln_files, desc="处理源代码文件"):
                    try:
                        with open(src_path, 'r', encoding='latin-1', errors='ignore') as f:
                            src_code = f.read()
                            
                        if not src_code.strip():
                            continue
                            
                        src_embedding = get_src_embedding(src_code, src_tokenizer, src_model, device)
                        src_embeddings[str(src_path)] = src_embedding
                    except Exception as e:
                        print(f"处理源代码文件出错 {src_path}: {e}")
                
                print(f"成功处理了 {len(src_embeddings)} 个源代码文件")
                
                # 对每个函数计算与所有源代码的相似度
                top_n = 5  # 为每个函数保留最相似的前5个结果
                all_function_results = []
                
                print("\n计算所有汇编函数与源代码的相似度...")
                for func in tqdm(asm_functions, desc="处理汇编函数"):
                    # 清理函数名称，提取原始名称
                    clean_name = func['name']
                    # 如果名称包含点号，取最后一部分（通常是原始函数名）
                    if '.' in clean_name:
                        clean_name = clean_name.split('.')[-1]
                    # 如果包含子函数标记，取主函数名
                    if '_sub_' in clean_name:
                        clean_name = clean_name.split('_sub_')[0]
                    
                    # 获取函数嵌入
                    asm_embedding = get_asm_embedding(func['asm_code'], asm_model, tokens, asm_mapper, device, use_position=True)
                    
                    # 计算与每个源代码文件的相似度
                    func_results = []
                    for src_path, src_embedding in src_embeddings.items():
                        similarity = calculate_similarity(asm_embedding, src_embedding)
                            
                            # 从文件名中提取漏洞类型
                        src_filename = os.path.basename(src_path)
                        cwe_match = re.search(r'CWE-?(\d+)', src_filename)
                        vuln_type = f"CWE-{cwe_match.group(1)}" if cwe_match else os.path.splitext(src_filename)[0]
                            
                            # 计算百分比形式的相似度
                        percentage = similarity* 100
                            
                        func_results.append({
                            'func_name': clean_name,
                            'func_offset': hex(func['offset']),
                            'func_size': func['size'],
                            'src_file': src_path,
                            'src_file_name': src_filename,
                                'vulnerability_type': vuln_type,
                                'similarity_score': similarity,
                                'similarity_percentage': percentage
                            })
                
                # 按相似度排序
                    func_results.sort(key=lambda x: x['similarity_score'], reverse=True)
                    
                    # 只保留前N个结果
                    top_func_results = func_results[:top_n]
                    if top_func_results:
                        all_function_results.append({
                            'func_name': clean_name,
                            'func_offset': hex(func['offset']),
                            'func_size': func['size'],
                            'top_matches': top_func_results
                        })
                
                # 如果找到了结果
                if all_function_results:
                    # 对所有函数结果按最高相似度排序
                    all_function_results.sort(key=lambda x: x['top_matches'][0]['similarity_score'], reverse=True)
                    
                    # 统计漏洞类型分布
                    vuln_type_counts = {}
                    all_func_vuln_results = []
                    
                    for func_result in all_function_results:
                        # 获取该函数最可能的漏洞类型（第一个匹配项）
                        top_match = func_result['top_matches'][0]
                        vuln_type = top_match['vulnerability_type']
                        
                        # 添加到函数级别的漏洞结果
                        all_func_vuln_results.append({
                            'func_name': func_result['func_name'],
                            'func_offset': func_result['func_offset'],
                            'func_size': func_result['func_size'],
                            'vulnerability_type': vuln_type,
                            'similarity_score': top_match['similarity_score'],
                            'similarity_percentage': top_match['similarity_percentage']
                        })
                        
                        # 统计漏洞类型
                        if vuln_type not in vuln_type_counts:
                            vuln_type_counts[vuln_type] = 1
                        else:
                            vuln_type_counts[vuln_type] += 1
                    
                    # 只保留函数名不以数字开头的函数
                    def not_start_with_digit(name):
                        return not name or not name[0].isdigit()
                    
                    filtered_func_vuln_results = [item for item in all_func_vuln_results if not_start_with_digit(item['func_name'])]
                    
                    # 按相似度排序函数级别的漏洞结果
                    filtered_func_vuln_results.sort(key=lambda x: x['similarity_score'], reverse=True)
                    
                    # 获取最可能的整体漏洞类型（出现最多的类型）
                    most_common_vuln_type = max(vuln_type_counts.items(), key=lambda x: x[1])[0]
                    
                    # 使用最高相似度的函数结果作为整体结果
                    overall_result = {
                        'vulnerability_type': most_common_vuln_type,
                        'similarity_score': filtered_func_vuln_results[0]['similarity_score'] if filtered_func_vuln_results else 0,
                        'similarity_percentage': filtered_func_vuln_results[0]['similarity_percentage'] if filtered_func_vuln_results else 0
                    }
                    
                    # 计算每种漏洞类型的百分比
                    total_functions = len(all_function_results)
                    vuln_type_percentages = {}
                    for vuln_type, count in vuln_type_counts.items():
                        percentage = (count / total_functions) * 100
                        vuln_type_percentages[vuln_type] = {
                            'count': count,
                            'percentage': percentage
                        }
                    
                    # 保存分析历史
                    analysis = AnalysisHistory.objects.create(
                        user=request.user,
                        analysis_type='vulnerability',
                        file1=request.FILES['file'],
                        similarity_score=overall_result['similarity_score'],
                        remarks=overall_result['vulnerability_type'],
                        top_results=json.dumps({
                            'overall_result': overall_result,
                            'function_results': filtered_func_vuln_results[:10],  # 保存前10个函数结果
                            'similarity_results': all_function_results[:10]  # 保存前10个详细结果
                        })
                    )
                    
                    messages.success(request, f'分析完成！以下是我们的分析结果，因为涉及反汇编，仅供参考。')
                    
                    return render(request, 'core/vulnerability_analysis.html', {
                        'form': form,
                        'overall_result': overall_result,
                        'function_results': filtered_func_vuln_results[:10],  # 显示前10个函数结果
                        'total_functions': len(all_function_results),
                        'total_files': len(vuln_files),
                        'vuln_type_counts': vuln_type_counts,
                        'vuln_type_percentages': vuln_type_percentages
                    })
                else:
                    messages.warning(request, '未找到相似的漏洞模式')
                    return render(request, 'core/vulnerability_analysis.html', {'form': form})
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                messages.error(request, f'分析过程中出现错误：{str(e)}')
            finally:
                # 清理临时文件
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    if 'temp_dir_asm' in locals() and os.path.exists(temp_dir_asm):
                        import shutil
                        shutil.rmtree(temp_dir_asm)
                except Exception as e:
                    print(f"清理临时文件时出错: {e}")
    else:
        form = VulnerabilityAnalysisForm()
    return render(request, 'core/vulnerability_analysis.html', {'form': form})

# 从bert_similarity.py导入的函数
def load_model(model_path):
    """加载训练好的模型"""
    print(f"正在加载模型 {model_path}...")
    device = 'cpu'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化BERT模型
    bert_model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
    
    # 加载保存的模型权重
    checkpoint = torch.load(model_path, map_location=device)
    bert_model.load_state_dict(checkpoint['bert_model_state_dict'])
    
    # 设置为评估模式
    bert_model.eval()
    
    return bert_model, device

def get_file_embedding(model, tokenizer, file_path, device):
    """获取文件的嵌入向量"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None
    
    # 设置参数
    max_length = 512
    stride = 256  # 滑动窗口的步长
    
    # 对代码进行分词
    tokens = tokenizer.tokenize(code)
    
    # 如果代码长度小于最大长度，直接处理
    if len(tokens) <= max_length:
        encoded = tokenizer(code, return_tensors='pt', truncation=True, padding=False, max_length=max_length)
        inputs = encoded.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
        
        return embedding
    
    # 对于长代码，使用滑动窗口处理
    print(f"文件 {file_path} 长度超过最大限制，使用滑动窗口处理")
    
    embeddings = []
    num_windows = (len(tokens) - max_length) // stride + 2
    
    for i in range(num_windows):
        start_idx = i * stride
        if start_idx >= len(tokens):
            break
        
        window_tokens = tokens[start_idx:start_idx + max_length]
        window_text = tokenizer.convert_tokens_to_string(window_tokens)
        
        encoded = tokenizer(window_text, return_tensors='pt', truncation=True, padding=False, max_length=max_length)
        inputs = encoded.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            window_embedding = outputs.last_hidden_state[:, 0, :]
            embeddings.append(window_embedding)
    
    if not embeddings:
        print(f"警告: 文件 {file_path} 未能生成有效嵌入")
        return None
    
    stacked_embeddings = torch.stack(embeddings, dim=0)
    final_embedding = torch.mean(stacked_embeddings, dim=0)
    
    return final_embedding

def calculate_similarity(embedding1, embedding2):
    """计算两个嵌入向量之间的余弦相似度"""
    embedding1_norm = nn.functional.normalize(embedding1, p=2, dim=1)
    embedding2_norm = nn.functional.normalize(embedding2, p=2, dim=1)
    
    similarity = torch.mm(embedding1_norm, embedding2_norm.T).item()
    
    return similarity

def extract_functions_from_asm(asm_file_path):
    """
    从汇编文件中提取所有函数及其内容
    返回字典，键为函数名，值为函数内容
    只提取以fcn.开头的函数
    """
    functions = {}
    
    try:
        # 先读取整个文件内容
        with open(asm_file_path, 'r', errors='replace') as f:
            content = f.read()
        
        # 使用正则表达式专门查找fcn.开头的函数
        import re
        # 匹配模式：fcn.数字: 后面跟着的内容，直到下一个fcn.或文件结束
        function_pattern = re.compile(r'(^|\n)(fcn\.\d+):(.*?)(?=\n(?:fcn\.\d+):|$)', re.DOTALL)
        function_matches = function_pattern.findall(content)
        
        for _, func_name, func_body in function_matches:
            func_name = func_name.strip()
            # 确保函数名有效且函数体不为空
            if func_name and len(func_body.strip()) > 0:
                functions[func_name] = f"{func_name}:\n{func_body}"
        
        # 如果没有找到任何函数，尝试更宽松的方式，但仍然只识别fcn.开头的
        if not functions:
            print("尝试使用更宽松的方式提取fcn.开头的函数...")
            lines = content.split('\n')
            current_func = None
            func_content = []
            
            for line in lines:
                # 检查是否是fcn.开头的函数定义行
                if ':' in line:
                    func_name = line.split(':', 1)[0].strip()
                    if func_name.startswith('fcn.'):
                        # 如果已有函数在收集中，保存它
                        if current_func and func_content:
                            functions[current_func] = '\n'.join(func_content)
                        
                        # 开始新函数
                        current_func = func_name
                        func_content = [line]
                        continue
                
                # 如果当前在函数内，继续收集内容
                if current_func:
                    func_content.append(line)
            
            # 保存最后一个函数
            if current_func and func_content:
                functions[current_func] = '\n'.join(func_content)
    
    except Exception as e:
        print(f"从汇编文件提取函数时出错: {str(e)}")
    
    # 调试信息
    if functions:
        print(f"成功从汇编文件中提取了 {len(functions)} 个fcn.开头的函数")
        print(f"函数名列表: {', '.join(list(functions.keys())[:5])}...等" if len(functions) > 5 else f"函数名列表: {', '.join(functions.keys())}")
    else:
        print("未能从汇编文件中提取任何fcn.开头的函数")
        # 输出文件中包含的fcn.行数量以供调试
        try:
            with open(asm_file_path, 'r', errors='replace') as f:
                fcn_count = sum(1 for line in f if 'fcn.' in line)
                print(f"文件中包含 {fcn_count} 行含有'fcn.'的内容")
                
                # 重置文件指针并显示一些包含fcn.的行作为示例
                f.seek(0)
                fcn_examples = []
                for i, line in enumerate(f):
                    if 'fcn.' in line and len(fcn_examples) < 5:
                        fcn_examples.append(f"行 {i+1}: {line.strip()}")
                if fcn_examples:
                    print("fcn.行示例:\n" + "\n".join(fcn_examples))
        except Exception as e:
            print(f"读取文件调试信息时出错: {e}")
    
    return functions

@login_required
def change_password_view(request):
    if request.method == 'POST':
        form = CustomPasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            # 更新会话，防止密码修改后被登出
            update_session_auth_hash(request, user)
            messages.success(request, '密码修改成功！')
            return redirect('dashboard')
        else:
            messages.error(request, '请修正下面的错误。')
    else:
        form = CustomPasswordChangeForm(request.user)
    return render(request, 'core/change_password.html', {'form': form})

@login_required
def delete_history_view(request, history_id):
    """删除分析历史记录"""
    history = get_object_or_404(AnalysisHistory, id=history_id, user=request.user)
    
    # 删除关联的文件
    try:
        if history.file1 and history.file1.storage.exists(history.file1.name):
            history.file1.delete(save=False)
        if history.file2 and history.file2.storage.exists(history.file2.name):
            history.file2.delete(save=False)
    except Exception as e:
        print(f"清理历史文件时出错: {e}")
    
    # 删除记录
    history.delete()
    messages.success(request, '历史记录已成功删除！')
    return redirect('dashboard')

@login_required
def analysis_detail_view(request, analysis_id):
    """显示分析历史的详细信息"""
    analysis = get_object_or_404(AnalysisHistory, id=analysis_id, user=request.user)
    
    context = {
        'analysis': analysis,
    }
    
    if analysis.analysis_type == 'vulnerability':
        # 漏洞分析详情
        try:
            # 尝试加载新格式的JSON数据（函数级别的分析结果）
            top_results = analysis.get_top_results()
            
            if isinstance(top_results, dict) and 'overall_result' in top_results and 'function_results' in top_results:
                # 新格式：函数级别的分析结果
                context['overall_result'] = top_results['overall_result']
                context['function_results'] = top_results['function_results']
                context['total_functions'] = len(top_results['function_results'])
                
                # 统计漏洞类型分布
                vuln_type_counts = {}
                for result in top_results['function_results']:
                    vuln_type = result['vulnerability_type']
                    if vuln_type not in vuln_type_counts:
                        vuln_type_counts[vuln_type] = 1
                    else:
                        vuln_type_counts[vuln_type] += 1
                
                # 计算每种漏洞类型的百分比
                total_vuln_funcs = len(top_results['function_results'])
                vuln_type_percentages = {}
                for vuln_type, count in vuln_type_counts.items():
                    percentage = (count / total_vuln_funcs) * 100
                    vuln_type_percentages[vuln_type] = {
                        'count': count,
                        'percentage': percentage
                    }
                
                context['vuln_type_counts'] = vuln_type_counts
                context['vuln_type_percentages'] = vuln_type_percentages
                
                return render(request, 'core/vulnerability_detail.html', context)
            else:
                # 旧格式：整体漏洞分析结果
                context['similarity_score'] = analysis.similarity_score
                context['vulnerability_type'] = analysis.remarks
                
                # 检查并转换为百分比形式（如果需要）
                formatted_results = []
                for item in top_results:
                    # 兼容不同的旧格式
                    if isinstance(item, tuple) and len(item) == 3:
                        # 已经包含百分比的格式
                        formatted_results.append(item)
                    elif isinstance(item, tuple) and len(item) == 2:
                        # 需要转换为百分比的格式
                        filename, score = item
                        percentage = (score + 1) * 50  # 将[-1,1]映射到[0,100]的百分比
                        formatted_results.append((filename, score, percentage))
                    elif isinstance(item, list) and len(item) >= 2:
                        # 列表格式
                        filename, score = item[0], item[1]
                        percentage = (score + 1) * 50
                        formatted_results.append((filename, score, percentage))
                
                context['top_results'] = formatted_results
                # 添加百分比形式的相似度
                context['similarity_percentage'] = (analysis.similarity_score + 1) * 50
        except Exception as e:
            # 处理JSON解析错误或其他异常
            print(f"分析历史数据解析错误: {e}")
            context['error'] = f"无法解析分析结果数据: {str(e)}"
            context['similarity_score'] = analysis.similarity_score
            context['vulnerability_type'] = analysis.remarks
            context['similarity_percentage'] = (analysis.similarity_score + 1) * 50
        
        return render(request, 'core/analysis_detail.html', context)
    else:
        # 二进制比对详情
        context['similarity_score'] = analysis.similarity_score
        context['similarity_percentage'] = (analysis.similarity_score + 1) * 50
        
        # 加载函数对相似度信息
        if analysis.top_results:
            try:
                top_pairs = json.loads(analysis.top_results)
                context['top_pairs'] = top_pairs
            except json.JSONDecodeError:
                context['top_pairs'] = []
        
        return render(request, 'core/analysis_detail.html', context)

# 管理员相关视图
def is_admin(user):
    """检查用户是否是管理员"""
    return user.is_authenticated and user.is_superuser

@login_required(login_url='login')
def admin_dashboard(request):
    """管理员仪表盘"""
    if not is_admin(request.user):
        messages.error(request, '您没有权限访问管理员页面')
        return redirect('dashboard')
    
    # 获取最近的分析记录
    recent_analyses = AnalysisHistory.objects.all().order_by('-created_at')[:10]
    # 获取所有用户
    from django.contrib.auth.models import User
    users = User.objects.filter(is_superuser=False)
    
    context = {
        'recent_analyses': recent_analyses,
        'users': users,
    }
    
    return render(request, 'core/admin/dashboard.html', context)

@login_required(login_url='login')
def admin_user_analyses(request, user_id):
    """查看指定用户的所有分析记录"""
    if not is_admin(request.user):
        messages.error(request, '您没有权限访问管理员页面')
        return redirect('dashboard')
    
    from django.contrib.auth.models import User
    user = get_object_or_404(User, id=user_id)
    analyses = AnalysisHistory.objects.filter(user=user).order_by('-created_at')
    
    context = {
        'target_user': user,
        'analyses': analyses,
    }
    
    return render(request, 'core/admin/user_analyses.html', context)

@login_required(login_url='login')
def admin_reset_user_password(request, user_id):
    """重置用户密码"""
    if not is_admin(request.user):
        messages.error(request, '您没有权限访问管理员页面')
        return redirect('dashboard')
    
    from django.contrib.auth.models import User
    user = get_object_or_404(User, id=user_id)
    
    if request.method == 'POST':
        new_password = request.POST.get('new_password')
        if new_password:
            user.set_password(new_password)
            user.save()
            messages.success(request, f'用户 {user.username} 的密码已成功重置')
            return redirect('admin_dashboard')
        else:
            messages.error(request, '请输入有效的新密码')
    
    return render(request, 'core/admin/reset_password.html', {'target_user': user})

@login_required(login_url='login')
def admin_upload_vulnerability_file(request):
    """上传新的漏洞文件"""
    if not is_admin(request.user):
        messages.error(request, '您没有权限访问管理员页面')
        return redirect('dashboard')
    
    from django.conf import settings
    import os
    
    # 获取脚本所在的根目录（asm2vec-pytorch）
    scripts_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    vuln_db_dir = os.path.join(scripts_root, 'data/src1/')
    
    if request.method == 'POST':
        vuln_file = request.FILES.get('vulnerability_file')
        vuln_type = request.POST.get('vulnerability_type')
        
        if vuln_file and vuln_type:
            # 确保目录存在
            os.makedirs(vuln_db_dir, exist_ok=True)
            
            # 构建文件名和路径
            file_name = f"{vuln_file.name}"
            file_path = os.path.join(vuln_db_dir, file_name)
            
            # 保存文件
            with open(file_path, 'wb+') as f:
                for chunk in vuln_file.chunks():
                    f.write(chunk)
            
            messages.success(request, f'漏洞文件 {file_name} 上传成功')
            return redirect('admin_dashboard')
        else:
            messages.error(request, '请提供有效的漏洞文件和类型')
    
    # 获取现有漏洞文件列表
    existing_files = []
    if os.path.exists(vuln_db_dir):
        for file in os.listdir(vuln_db_dir):
            if file.endswith(('.c', '.cpp', '.cc', '.h', '.s')):
                file_path = os.path.join(vuln_db_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read(200)
                except Exception:
                    content = ''
                existing_files.append({'name': file, 'content': content})
    
    return render(request, 'core/admin/upload_vulnerability.html', {'existing_files': existing_files})

@login_required(login_url='login')
def admin_update_model(request):
    """更新分析模型"""
    if not is_admin(request.user):
        messages.error(request, '您没有权限访问管理员页面')
        return redirect('dashboard')
    
    from django.conf import settings
    import os
    from .models import SystemConfig
    
    # 获取脚本所在的根目录（asm2vec-pytorch）
    scripts_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 获取现有模型列表
    model_files = []
    for file in os.listdir(scripts_root):
        if file.endswith('.pt'):
            model_files.append(file)
    
    # 获取当前使用的模型
    active_binary_model = SystemConfig.get_value('binary_compare_model', 'model318.pt')
    active_vuln_asm2vec_model = SystemConfig.get_value('vulnerability_asm2vec_model', 'model420.pt')
    active_vuln_mapping_model = SystemConfig.get_value('vulnerability_mapping_model', 'model_asm2vec5012.pt')
    
    if request.method == 'POST':
        action = request.POST.get('action')
        # 处理模型上传
        if request.FILES.get('model_file') and request.POST.get('model_name'):
            model_file = request.FILES.get('model_file')
            model_name = request.POST.get('model_name')
            if not model_name.endswith('.pt'):
                model_name += '.pt'
            file_path = os.path.join(scripts_root, model_name)
            with open(file_path, 'wb+') as f:
                for chunk in model_file.chunks():
                    f.write(chunk)
            messages.success(request, f'模型 {model_name} 上传成功')
            return redirect('admin_update_model')
        # 设置二进制比对模型
        elif action == 'set_binary_model' and request.POST.get('model_file'):
            model_file = request.POST.get('model_file')
            if model_file in model_files:
                SystemConfig.set_value('binary_compare_model', model_file, '用于二进制比对的模型文件名')
                messages.success(request, f'已将 {model_file} 设置为二进制比对模型')
                return redirect('admin_update_model')
            else:
                messages.error(request, f'模型文件 {model_file} 不存在')
        # 设置漏洞ASM2VEC模型
        elif action == 'set_vuln_asm2vec_model' and request.POST.get('model_file'):
            model_file = request.POST.get('model_file')
            if model_file in model_files:
                SystemConfig.set_value('vulnerability_asm2vec_model', model_file, '用于漏洞分析的ASM2VEC模型文件名')
                messages.success(request, f'已将 {model_file} 设置为漏洞ASM2VEC模型')
                return redirect('admin_update_model')
            else:
                messages.error(request, f'模型文件 {model_file} 不存在')
        # 设置漏洞映射模型
        elif action == 'set_vuln_mapping_model' and request.POST.get('model_file'):
            model_file = request.POST.get('model_file')
            if model_file in model_files:
                SystemConfig.set_value('vulnerability_mapping_model', model_file, '用于漏洞分析的映射模型文件名')
                messages.success(request, f'已将 {model_file} 设置为漏洞映射模型')
                return redirect('admin_update_model')
            else:
                messages.error(request, f'模型文件 {model_file} 不存在')
        # 删除未使用的模型
        elif action == 'delete_model' and request.POST.get('model_file'):
            model_file = request.POST.get('model_file')
            if model_file not in [active_binary_model, active_vuln_asm2vec_model, active_vuln_mapping_model]:
                model_path = os.path.join(scripts_root, model_file)
                if os.path.exists(model_path):
                    try:
                        os.remove(model_path)
                        messages.success(request, f'模型 {model_file} 已成功删除')
                    except Exception as e:
                        messages.error(request, f'删除模型 {model_file} 时出错: {str(e)}')
                else:
                    messages.error(request, f'模型文件 {model_file} 不存在')
            else:
                messages.error(request, f'无法删除正在使用中的模型 {model_file}')
            return redirect('admin_update_model')
        else:
            messages.error(request, '请提供有效的模型文件和名称')
    
    return render(request, 'core/admin/update_model.html', {
        'models': model_files,
        'active_binary_model': active_binary_model,
        'active_vuln_asm2vec_model': active_vuln_asm2vec_model,
        'active_vuln_mapping_model': active_vuln_mapping_model
    })

@login_required(login_url='login')
def admin_delete_vulnerability_file(request, file_name):
    """删除漏洞文件"""
    if not is_admin(request.user):
        messages.error(request, '您没有权限访问管理员页面')
        return redirect('dashboard')
    
    from django.conf import settings
    import os
    
    # 获取脚本所在的根目录（asm2vec-pytorch）
    scripts_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    vuln_db_dir = os.path.join(scripts_root, 'data/asm_vul/')
    
    file_path = os.path.join(vuln_db_dir, file_name)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        messages.success(request, f'漏洞文件 {file_name} 已成功删除')
    else:
        messages.error(request, f'漏洞文件 {file_name} 不存在')
    
    return redirect('admin_upload_vulnerability')

def disassemble_binary(file_path):
    """
    使用radare2反汇编二进制文件
    返回反汇编后的文本内容
    """
    try:
        # 打开二进制文件
        r2 = r2pipe.open(file_path)
        
        # 初始化radare2
        r2.cmd('aaa')  # 进行深入分析
        
        # 获取所有函数
        functions = r2.cmdj('aflj')  # 获取JSON格式的函数列表
        
        if not functions:
            print("未找到任何函数")
            return None
        
        # 存储所有反汇编内容
        disassembly = []
        
        # 遍历每个函数并获取其反汇编代码
        for func in functions:
            func_name = func.get('name', '')
            if not func_name.startswith('fcn.'):
                continue
                
            # 获取函数的反汇编代码
            disasm = r2.cmd(f'pdf @ {func_name}')
            if disasm:
                disassembly.append(f"\n{disasm}")
        
        # 关闭radare2
        r2.quit()
        
        if not disassembly:
            print("未能获取任何反汇编代码")
            return None
            
        return '\n'.join(disassembly)
        
    except Exception as e:
        print(f"反汇编过程中出错: {str(e)}")
        return None

@csrf_exempt
def get_cwe_info(request, cwe_number):
    """获取指定CWE编号的详细信息"""
    try:
        # 构建CWE信息URL
        url = f"https://cwe.mitre.org/data/definitions/{cwe_number}.html"
        
        # 发送HTTP请求获取页面内容
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return JsonResponse({
                'success': False,
                'error': f"无法获取CWE-{cwe_number}的信息，状态码: {response.status_code}"
            })
        
        # 使用正则表达式提取标题和描述
        html_content = response.text
        
        # 提取标题
        title_match = re.search(r'<title>(.*?)</title>', html_content)
        title = title_match.group(1) if title_match else f"CWE-{cwe_number}"
        title = title.replace(f"CWE-{cwe_number}: ", "")
        
        # 默认描述文本
        description = "没有找到详细描述"
        
        # 从页面上的具体图示中，尝试匹配Description部分
        # 1. 首先尝试匹配<h2>Description</h2>下方的内容
        desc_header_match = re.search(r'<h2[^>]*>Description</h2>(.*?)(?:<h2[^>]*>|<div class="collapse">)', html_content, re.DOTALL)
        if desc_header_match:
            # 提取这个部分中的段落文本
            desc_content = desc_header_match.group(1)
            p_content_match = re.search(r'<p[^>]*>(.*?)</p>', desc_content, re.DOTALL)
            if p_content_match:
                description = p_content_match.group(1)
        
        # 2. 尝试直接匹配包含"The product copies"的段落（根据页面截图）
        if description == "没有找到详细描述":
            product_copies_match = re.search(r'<p[^>]*>\s*The product copies an input buffer to an output buffer without verifying[^<]*</p>', html_content, re.DOTALL)
            if product_copies_match:
                description = product_copies_match.group(0)
                description = re.sub(r'</?p[^>]*>', '', description)
        
        # 3. 尝试匹配Description部分下的任何文本内容
        if description == "没有找到详细描述":
            desc_match = re.search(r'Description.*?<div[^>]*>(.*?)</div>', html_content, re.DOTALL | re.IGNORECASE)
            if desc_match:
                description = desc_match.group(1)
        
        # 4. 最后的尝试：匹配页面中任何包含"buffer overflow"相关描述的段落
        if description == "没有找到详细描述":
            buffer_match = re.search(r'<p[^>]*>([^<]*(?:buffer|overflow|copies)[^<]*)</p>', html_content, re.IGNORECASE)
            if buffer_match:
                description = buffer_match.group(1)
            
            # 如果CWE-120，直接硬编码描述（因为我们知道具体内容）
            if cwe_number == "120":
                description = "The product copies an input buffer to an output buffer without verifying that the size of the input buffer is less than the size of the output buffer, leading to a buffer overflow."
        
        # 清理HTML标签并格式化文本
        description = re.sub(r'<[^>]+>', ' ', description).strip()
        description = re.sub(r'\s+', ' ', description)
        
        return JsonResponse({
            'success': True,
            'cwe_number': cwe_number,
            'title': title,
            'description': description,
            'url': url
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f"获取CWE信息时出错: {str(e)}"
        })
