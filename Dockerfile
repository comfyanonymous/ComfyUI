FROM python:3.12-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=on

# 安装必要的软件包
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 克隆自定义节点
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git /app/custom_nodes/ComfyUI-Manager

# 安装依赖
COPY requirements.txt requirements.txt
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# 寻找所有目录下的requirements.txt文件并安装
RUN find . -name requirements.txt -exec pip install -r {} \;

# 拷贝当前目录下的所有文件到工作目录
COPY . .

# 执行主程序
CMD ["python3", "main.py"]
