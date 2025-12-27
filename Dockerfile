# 替换为自己的python镜像地址
FROM registry.cn-hangzhou.aliyuncs.com/wxa/python:3.12
# 设置工作目录
WORKDIR /app
# 复制 requirements.txt 和安装依赖
COPY requirements.txt /app/requirements.txt
# 安装依赖，指定阿里镜像源
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r /app/requirements.txt
# 复制源代码到容器内
COPY . /app
# 设置容器的默认命令
CMD ["python", "main.py"]