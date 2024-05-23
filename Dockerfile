FROM harbor.intra.ke.com/aistudio/serving/public/arch-inference-serving_dimage:master_202405090948
LABEL authors="shenenqing001"

ARG commit=main

ENV DEBIAN_FRONTEND=noninteractive LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

ARG CONDA_VERSION=py310_23.3.1-0


RUN pip install --upgrade pip -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install opencv-python -i  https://pypi.tuna.tsinghua.edu.cn/simple


RUN pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install torchsde -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install transformers>=4.25.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install safetensors>=0.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install aiohttp -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install psutil -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install kornia>=0.7.1 -i https://pypi.tuna.tsinghua.edu.cn/simple


RUN rm -rf /workspace
RUN mkdir -p /workspace
RUN mkdir -p /data0/www/applogs/
WORKDIR /workspace
COPY . /workspace/ComfyUI

RUN find /workspace/ComfyUI/custom_nodes -type f -name 'requirements.txt' -exec pip install -r {} \;

ENV TZ=Asia/Shanghai
ENTRYPOINT python /workspace/ComfyUI/main.py