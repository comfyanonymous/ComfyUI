FROM python:3.10-slim


ENV CLI_ARGS=""

COPY . /opt/comfy_ui
WORKDIR /opt/comfy_ui

RUN PIP_NO_CACHE_DIR=1 pip install --extra-index-url https://download.pytorch.org/whl/cu117 \
torch==1.13.1 \
torchvision \
torchaudio \
xformers
RUN pip install -r requirements.txt

CMD python main.py ${CLI_ARGS}
