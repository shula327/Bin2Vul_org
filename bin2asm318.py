#!/usr/bin/env python3
import re
import os
import click
import r2pipe
import hashlib
from pathlib import Path

def sha3(data):
    return hashlib.sha3_256(data.encode()).hexdigest()

def validEXE(filename):
    elf_magic = bytes.fromhex('7f454c46')  # ELF 魔数
    pe_magic = b'MZ'  # PE (Windows EXE) 魔数
    
    with open(filename, 'rb') as f:
        header = f.read(4)
        if header == elf_magic:
            return 'elf'
        elif header.startswith(pe_magic):
            return 'pe'  # 识别 PE 文件
        else:
            return None

def normalize(opcode):
    opcode = opcode.replace(' - ', ' + ')
    opcode = re.sub(r'0x[0-9a-f]+', 'CONST', opcode)
    opcode = re.sub(r'\*[0-9]', '*CONST', opcode)
    opcode = re.sub(r' [0-9]', ' CONST', opcode)
    return opcode

def fn2asm(pdf, minlen):
    # 检查函数有效性
    if pdf is None:
        return
    if len(pdf['ops']) < minlen:
        return
    if 'invalid' in [op['type'] for op in pdf['ops']]:
        return

    ops = pdf['ops']

    # 设置标签
    labels, scope = {}, [op['offset'] for op in ops]
    assert(None not in scope)
    for i, op in enumerate(ops):
        if op.get('jump') in scope:
            labels.setdefault(op.get('jump'), i)
    
    # 生成输出
    output = ''
    for op in ops:
        # 添加标签
        if labels.get(op.get('offset')) is not None:
            output += f'LABEL{labels[op["offset"]]}:\n'
        # 添加指令
        if labels.get(op.get('jump')) is not None:
            output += f' {op["type"]} LABEL{labels[op["jump"]]}\n'
        else:
            output += f' {normalize(op["opcode"])}\n'

    return output

def bin2asm(filename, opath, minlen):
    # 检查文件有效性
    file_type = validEXE(filename)
    if file_type is None:
        print(f"不支持的文件类型: {filename}")
        return 0
    
    r = r2pipe.open(str(filename))
    r.cmd('aaaa')  # 分析整个二进制文件

    # 使用输入文件的基本名称作为汇编文件的名称
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # 创建一个包含所有函数的汇编文件
    combined_asm_path = opath / f"{base_name}.s"
    combined_asm_content = f''' .file {os.path.basename(filename)}
 .type {file_type}
 .source {filename}

'''

    # 获取函数列表
    functions = r.cmdj('aflj')
    if not functions:
        print(f"未找到任何函数: {filename}")
        return 0

    count = 0
    for fn in functions:
        # 处理PE文件中的特殊情况
        if file_type == 'pe':
            # 确保我们有正确的偏移量信息
            r.cmd(f's {fn["offset"]}')
        else:
            r.cmd(f's {fn["offset"]}')  # 设置当前函数的偏移量
            
        pdf_data = r.cmdj('pdfj')
        asm = fn2asm(pdf_data, minlen)  # 生成汇编代码
        
        if asm:
            # 为每个函数创建一个段落，包含函数名和偏移量信息
            function_asm = f'''
;=======================================
; function: {fn["name"]}
;=======================================
{fn["name"]}:
{asm}

'''
            # 将函数汇编代码添加到合并文件内容中
            combined_asm_content += function_asm
            count += 1

    # 只有在找到有效函数时才保存文件
    if count > 0:
        with open(combined_asm_path, 'w') as f:
            f.write(combined_asm_content)
        print(f'[+] 已保存合并的汇编文件: {combined_asm_path}')

    print(f'[+] {filename} 处理完成，提取的汇编函数数量: {count}')
    return count

@click.command()
@click.option('-i', '--input', 'ipath', default='bin', help='input directory / file', required=True)
@click.option('-o', '--output', 'opath', default='asm3', help='output directory for assembly files')
@click.option('-l', '--len', 'minlen', default=20, help='ignore assembly code with instructions amount smaller than minlen')
def cli(ipath, opath, minlen):
    '''
    Extract assembly functions from binary executable (ELF and PE formats)
    '''
    ipath = Path(ipath)
    opath = Path(opath)

    # 创建输出目录
    if not os.path.exists(opath):
        os.mkdir(opath)

    fcount, bcount = 0, 0

    # 处理目录
    if os.path.isdir(ipath):
        for f in os.listdir(ipath):
            if not os.path.islink(ipath / f) and not os.path.isdir(ipath / f):
                fcount += bin2asm(ipath / f, opath, minlen)
                bcount += 1
    # 处理文件
    elif os.path.exists(ipath):
        fcount += bin2asm(ipath, opath, minlen)
        bcount += 1
    else:
        print(f'[Error] No such file or directory: {ipath}')

    print(f'[+] Total scan binary: {bcount} => Total generated assembly functions: {fcount}')

if __name__ == '__main__':
    cli()