#!/bin/bash

# 定义工作目录
WORKING_DIR="./custom_nodes"

# 检查工作目录是否存在
if [ ! -d "$WORKING_DIR" ]; then
  echo "目录 $WORKING_DIR 不存在."
  exit 1
fi

# 遍历 custom_nodes 目录下的所有子目录
for dir in "$WORKING_DIR"/*/; do
  # 检查是否是目录
  if [ -d "$dir" ]; then
    echo "进入目录: $dir"
    cd "$dir"

    # 执行 git pull
    echo "执行 git pull"
    git pull

    # 检查是否存在 requirements.txt 文件
    if [ -f "requirements.txt" ]; then
      echo "找到 requirements.txt，执行 pip install"
      pip install -r requirements.txt
    else
      echo "没有找到 requirements.txt，跳过 pip install"
    fi

    # 返回上级目录
    cd - > /dev/null
  fi
done

echo "所有目录已处理完毕."
