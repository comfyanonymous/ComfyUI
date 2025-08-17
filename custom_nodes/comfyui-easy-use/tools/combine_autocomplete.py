# 自定义提示词自动补全工具

import os, sys
import glob
import shutil

output_file = None
cwd_path = os.path.dirname(os.path.realpath(__file__))
pyssss_path = os.path.join(cwd_path, "..", "ComfyUI-Custom-Scripts", "user")
combine_folder = os.path.join(cwd_path, "autocomplete")

def backup_autocomplete():
    bak_file = os.path.join(pyssss_path, "autocomplete.txt.bak")
    if os.path.exists(bak_file):
        pass
    elif os.path.exists(output_file):
        shutil.copy(output_file, bak_file)

def combine_autocomplete():
    if os.path.exists(combine_folder):
        pass
    else:
        os.mkdir(combine_folder)
    if os.path.exists(pyssss_path):
        output_file = os.path.join(pyssss_path, "autocomplete.txt")
        # 遍历 combine 目录下的所有 txt 文件，读取内容并合并
        merged_content = ''
        for file_path in glob.glob(os.path.join(combine_folder, '*.txt')):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                try:
                    file_content = file.read()
                    merged_content += file_content + '\n'
                except UnicodeDecodeError:
                    pass
        if merged_content != '':
            # 将合并的内容写入目标文件 autocomplete.txt，并指定编码为 utf-8
            with open(output_file, 'w', encoding='utf-8') as target_file:
                target_file.write(merged_content)

if __name__ == "__main__":
    arg = sys.argv[0]
    if 'combine_autocomplete' in arg:
        arg = sys.argv[1]
    if arg == 'backup':
        backup_autocomplete()
    elif arg == 'combine':
        combine_autocomplete()
    else:
        print("Usage: python combine_autocomplete.py [backup|combine]")
        sys.exit(1)