#!/bin/bash

requirements_txt="$(dirname "$0")/requirements.txt"
requirements_repair_txt="$(dirname "$0")/repair_dependency_list.txt"
python_exec="../../../python_embeded/python.exe"
aki_python_exec="../../python/python.exe"

echo "Installing EasyUse Requirements..."

if [ -f "$python_exec" ]; then
    echo "Installing with ComfyUI Portable"
    "$python_exec" -s -m pip install -r "$requirements_txt"
elif [ -f "$aki_python_exec" ]; then
    echo "Installing with ComfyUI Aki"
    "$aki_python_exec" -s -m pip install -r "$requirements_txt"
    while IFS= read -r line; do
        "$aki_python_exec" -s -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "$line"
    done < "$requirements_repair_txt"
else
    echo "Installing with system Python"
    pip install -r "$requirements_txt"
fi

read -p "Press any key to continue..."