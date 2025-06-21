import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import click
import asm2vec
import re
import numpy as np
import uuid

def cosine_similarity(v1, v2):
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()

def split_functions(asm_file):
    """Split an assembly file into separate functions"""
    with open(asm_file, 'r') as f:
        content = f.read()
    
    # 使用正则表达式查找函数区块
    # 查找模式：以function开头的注释行以及随后的代码，直到下一个function或文件结束
    function_pattern = r';=+\s*;\s*function:\s*([\w\.]+)\s*;=+\s*(.*?)(?=;=+|\Z)'
    matches = re.findall(function_pattern, content, re.DOTALL)
    
    functions = {}
    for func_name, func_body in matches:
        # 如果函数体过短或全是空白行，跳过它
        lines = [line.strip() for line in func_body.strip().split('\n') if line.strip()]
        if len(lines) <= 2:  # 跳过太短的函数
            continue
        functions[func_name] = func_body.strip()
    
    return functions

def write_function_to_temp(func_body, temp_dir="temp_funcs", suffix=""):
    """将单个函数写入临时文件，使用uuid确保文件名唯一"""
    os.makedirs(temp_dir, exist_ok=True)
    # 添加suffix和uuid确保唯一性
    unique_id = str(uuid.uuid4())[:8]
    temp_file = os.path.join(temp_dir, f"func_{hash(func_body) & 0xFFFFFFFF}_{suffix}_{unique_id}.s")
    with open(temp_file, 'w') as f:
        f.write(func_body)
    return temp_file

def safe_remove(file_path):
    """安全删除文件，忽略文件不存在的错误"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        pass

@click.command()
@click.option('-i1', '--input1', default='data/asm2/4b.s', help='First assembly file with multiple functions', required=True)
@click.option('-i2', '--input2', default='data/asm2/4b.s', help='Second assembly file with multiple functions', required=True)
@click.option('-m', '--model', default='model407.pt', help='model path', required=True)
@click.option('-e', '--epochs', default=5, help='training epochs', show_default=True)
@click.option('-c', '--device', default='auto', help='hardware device to be used: cpu / cuda / auto', show_default=True)
@click.option('-lr', '--learning-rate', 'lr', default=0.001, help="learning rate", show_default=True)
@click.option('-t', '--threshold', default=0, help='Minimum function size threshold', show_default=True)
@click.option('-q', '--quiet', is_flag=True, help='Only output final similarity value', show_default=True)
def cli(input1, input2, model, epochs, device, lr, threshold, quiet):
    # 设置为安静模式
    if not quiet:
        print(f"Processing files: {input1}, {input2}")
    
    # 判断是否是自相似度（比较同一个文件）
    is_self_similarity = os.path.abspath(input1) == os.path.abspath(input2)
    if not quiet and is_self_similarity:
        print("Self-similarity analysis: Same file will be compared against itself")
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not quiet:
        print(f"Using device: {device}")
    
    # 创建临时目录
    temp_dir = "temp_funcs"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 加载模型
        if not os.path.exists(model):
            print(f"ERROR: Model file does not exist: {model}")
            sys.exit(1)
        
        model, tokens = asm2vec.utils.load_model(model, device=device)
        if not quiet:
            print("Successfully loaded model and tokens")
        
        # 分割函数
        functions1 = split_functions(input1)
        functions2 = split_functions(input2)
        
        if not quiet:
            print(f"File 1 contains {len(functions1)} functions")
            print(f"File 2 contains {len(functions2)} functions")
        
        # 如果没有函数，退出
        if not functions1 or not functions2:
            print("ERROR: No valid functions found in one or both files")
            sys.exit(1)
        
        # 计算所有函数对之间的相似度
        similarities = []
        pair_details = []
        
        # 为了避免创建太多临时文件，我们每次比较一对函数
        for func1_name, func1_body in functions1.items():
            for func2_name, func2_body in functions2.items():
                # 标记是否为相同名称的函数
                is_same_function = is_self_similarity and func1_name == func2_name
                
                if not quiet:
                    if is_same_function:
                        print(f"Comparing same function: {func1_name}")
                    else:
                        print(f"Comparing {func1_name} with {func2_name}")
                
                # 写入临时文件，添加不同的后缀确保唯一性
                temp_file1 = write_function_to_temp(func1_body, temp_dir, suffix="file1")
                temp_file2 = write_function_to_temp(func2_body, temp_dir, suffix="file2")
                
                try:
                    # 加载这两个函数
                    funcs, tokens_new = asm2vec.utils.load_data([temp_file1, temp_file2])
                    
                    # 如果无法加载函数，则跳过
                    if len(funcs) < 2:
                        if not quiet:
                            print(f"Warning: Could not load functions properly, skipping this pair")
                        continue
                    
                    # 更新tokens和模型
                    tokens.update(tokens_new)
                    model.update(2, tokens.size())
                    model = model.to(device)
                    
                    # 训练
                    model = asm2vec.utils.train(
                        funcs,
                        tokens,
                        model=model,
                        epochs=epochs,
                        device=device,
                        mode='test',
                        learning_rate=lr
                    )
                    
                    # 计算相似度
                    v1, v2 = model.to('cpu').embeddings_f(torch.tensor([0, 1]))
                    sim = cosine_similarity(v1, v2)
                    similarities.append(sim)
                    pair_details.append((func1_name, func2_name, sim, is_same_function))
                    
                    if not quiet:
                        if is_same_function:
                            print(f"Similarity of {func1_name} with itself: {sim:.6f}")
                        else:
                            print(f"Similarity between {func1_name} and {func2_name}: {sim:.6f}")
                    
                except Exception as e:
                    if not quiet:
                        print(f"Error processing function pair {func1_name} - {func2_name}: {e}")
                finally:
                    # 安全删除临时文件
                    safe_remove(temp_file1)
                    safe_remove(temp_file2)
        
        # 计算均值
        if similarities:
            # 基本平均相似度
            avg_similarity = np.mean(similarities)
            
            # 对于自相似度分析，计算相同名称的函数相似度平均值
            if is_self_similarity:
                same_name_sims = [sim for func1, func2, sim, is_same_func in pair_details if is_same_func]
                diff_name_sims = [sim for func1, func2, sim, is_same_func in pair_details if not is_same_func]
                
                if same_name_sims:
                    # 在自相似度模式下，使用相同名称函数的平均相似度作为最终结果
                    self_avg_similarity = np.mean(same_name_sims)
                    # 只输出最终的相似度值（在自相似度模式下使用self_avg_similarity）
                    print(f"cosine similarity : {self_avg_similarity:.6f}")
                else:
                    # 如果没有相同名称的函数，使用一般平均值
                    print(f"cosine similarity : {avg_similarity:.6f}")
            else:
                # 非自相似度分析，使用一般平均值
                print(f"cosine similarity : {avg_similarity:.6f}")
            
            # 即使在安静模式下也打印前5对最相似的函数
            print("\nTop 5 most similar function pairs:")
            sorted_pairs = sorted(pair_details, key=lambda x: x[2], reverse=True)
            for i, (func1, func2, sim, is_same_func) in enumerate(sorted_pairs[:5]):
                if is_same_func:
                    print(f"{func1} (self): {sim:.6f}")
                else:
                    print(f"{func1} - {func2}: {sim:.6f}")
            
            if not quiet:
                print(f"Total function pairs compared: {len(similarities)}")
                
                # 打印所有函数对的相似度，按相似度排序
                print("\nDetailed similarity results (sorted):")
                for func1, func2, sim, is_same_func in sorted_pairs:
                    if is_same_func:
                        print(f"{func1} (self): {sim:.6f}")
                    else:
                        print(f"{func1} - {func2}: {sim:.6f}")
            
                # 计算并打印自相似度统计 - 仅在非安静模式下显示
                if is_self_similarity:
                    if same_name_sims:
                        print(f"\nAverage self-similarity of functions: {np.mean(same_name_sims):.6f}")
                        print(f"Number of self-compared functions: {len(same_name_sims)}")
                        
                        # 打印自相似度的分布
                        same_name_sims.sort(reverse=True)
                        if len(same_name_sims) > 1:
                            print(f"Highest self-similarity: {same_name_sims[0]:.6f}")
                            print(f"Lowest self-similarity: {same_name_sims[-1]:.6f}")
                    
                    if diff_name_sims:
                        print(f"Average similarity between different functions: {np.mean(diff_name_sims):.6f}")
                        print(f"Number of different function pairs: {len(diff_name_sims)}")
        else:
            print("ERROR: No valid function pairs were compared")
            
    except Exception as e:
        print(f"ERROR: {e}")
        if not quiet:
            import traceback
            traceback.print_exc()
    finally:
        # 清理临时目录
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            if not quiet:
                print(f"Warning: Could not clean up temporary directory: {e}")

if __name__ == '__main__':
    # 默认添加安静模式
    sys.argv.append('--quiet')
    cli()