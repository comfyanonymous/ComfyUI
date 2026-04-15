FROM docker.io/nvidia/cuda:13.0.3-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update

RUN apt install -y python3.12 python3.12-venv
RUN python3.12 -m ensurepip --upgrade
RUN python3.12 -m pip install --upgrade pip

WORKDIR /app

RUN apt install -y git libgl1 libglib2.0-0

RUN git clone https://github.com/comfyanonymous/ComfyUI.git .

RUN git clone https://github.com/Comfy-Org/ComfyUI-Manager ./custom_nodes/comfyui‑manager

RUN python3.12 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130

RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

RUN python3.12 -m pip install -r ./custom_nodes/comfyui‑manager/requirements.txt

EXPOSE 8188

CMD ["python3.12", "main.py", "--listen", "0.0.0.0", "--enable-manager"]
