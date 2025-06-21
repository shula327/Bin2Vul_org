#!/usr/bin/env python3
import subprocess
from pathlib import Path
import argparse
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 定义参数

def convert_to_asm(input_file, output_dir, minlen):
    """调用 bin2asms.py 将二进制文件转换为汇编代码"""
    try:
        # print(f"Converting {input_file} to ASM in {output_dir} with min length {minlen}")  # 添加调试信息
        result = subprocess.run(['python', 'bin2asm318.py', 
                               '-i', str(input_file), 
                               '-o', str(output_dir), 
                               '-l', str(minlen)],
                              capture_output=True,
                              text=True,
                              #encoding='utf-8',  # 明确指定编码
                              errors='replace',  # 替换无法解码的字符
                              check=True)
        # print(f"Conversion stdout: {result.stdout}")  # 添加调试信息
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting to ASM: {e.stderr}", file=sys.stderr)
        print(f"Command: {' '.join(e.cmd)}")  # 打印出错的命令
        return False

def analyze_similarity(asm_file1, asm_file2, model, epochs, device, lr):
    """调用 test408_debug.py 分析两个汇编文件的相似度"""
    try:
        # Check if the files exist
        if not os.path.exists(asm_file1):
            print(f"ERROR: ASM file 1 does not exist: {asm_file1}")
            return None
        if not os.path.exists(asm_file2):
            print(f"ERROR: ASM file 2 does not exist: {asm_file2}")
            return None
        if not os.path.exists(model):
            print(f"ERROR: Model file does not exist: {model}")
            return None
        
        # Use test408_debug.py with quiet mode
        result = subprocess.run(['python', 'test408_debug.py',
                               '-i1', str(asm_file1),
                               '-i2', str(asm_file2),
                               '-m', model,
                               '-e', str(epochs),
                               '-c', device,
                               '-lr', str(lr),
                               #'-q'
                               ],  # 添加安静模式
                              encoding='utf-8',
                              capture_output=True,
                              errors='replace',
                              text=True,
                              check=False)
        
        # Check if the command succeeded
        if result.returncode != 0:
            print(f"Error: test408_debug.py exited with code {result.returncode}")
            return None
        
        # 直接输出结果，包含相似度和前5个最相似函数对
        print(result.stdout.strip())
        return 0
        
    except Exception as e:
        print(f"Unexpected error analyzing similarity: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='二进制代码相似度分析')
    parser.add_argument('-i1', '--input1', default='data/bin/4.exe', help='第一个二进制文件路径')
    parser.add_argument('-i2', '--input2', default='data/bin/4.exe', help='第二个二进制文件路径')
    parser.add_argument('-o', '--output', default='data/asm2', help='输出目录')
    parser.add_argument('-l', '--minlen', type=int, default=0, help='最小指令长度')
    parser.add_argument('-m', '--model', default='model407.pt', help='模型路径')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='训练周期')
    parser.add_argument('-c', '--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='学习率')
    
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model file does not exist: {args.model}")
        sys.exit(1)
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # 转换第一个二进制文件
    if not convert_to_asm(args.input1, output_path, args.minlen):
        print("Failed to convert first binary file to ASM", file=sys.stderr)
        sys.exit(1)

    # 转换第二个二进制文件
    if not convert_to_asm(args.input2, output_path, args.minlen):
        print("Failed to convert second binary file to ASM", file=sys.stderr)
        sys.exit(1)

    # 生成的汇编文件名
    asm_file1 = output_path / f"{Path(args.input1).stem}.s"
    asm_file2 = output_path / f"{Path(args.input2).stem}.s"

    # 检查文件是否存在
    if not asm_file1.exists():
        print(f"ERROR: ASM file 1 does not exist: {asm_file1}")
        sys.exit(1)
    
    if not asm_file2.exists():
        print(f"ERROR: ASM file 2 does not exist: {asm_file2}")
        sys.exit(1)

    # 分析相似度
    similarity_score = analyze_similarity(asm_file1, asm_file2, args.model, 
                                       args.epochs, args.device, args.learning_rate)
    
    if similarity_score is not None:
        # 只输出相似度得分，便于其他程序处理
        print(f"{similarity_score}")
        sys.exit(0)
    else:
        print("Failed to calculate similarity score", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 