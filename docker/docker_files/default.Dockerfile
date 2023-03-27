FROM python:3.10-slim


WORKDIR /opt/comfy_ui


RUN PIP_NO_CACHE_DIR=1 pip install --extra-index-url https://download.pytorch.org/whl/cu117 \
torch==1.13.1+cu117 \
torchvision \
torchaudio \
xformers \
triton

COPY . /opt/comfy_ui
RUN pip install -r requirements.txt


ENV CLI_ARGS=""
CMD python main.py ${CLI_ARGS}
