FROM python:3.12.5-bookworm

RUN pip install jinja2 numpy
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
#RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN apt update -y
RUN apt install -y git
RUN git clone https://github.com/comfyanonymous/ComfyUI.git
RUN apt install -y cargo cmake g++ gcc python3-dev musl-dev make nasm
RUN pip install -r /ComfyUI/requirements.txt

RUN apt remove -y git cargo cmake g++ gcc python3-dev musl-dev make nasm
RUN apt autoremove -y && apt clean -y

RUN chmod +x /ComfyUI/main.py

ENTRYPOINT ["python", "/ComfyUI/main.py", "--listen"]
