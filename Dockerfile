FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

RUN apt update

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev libgl1-mesa-glx

ENV HOME="/root"
WORKDIR ${HOME}
RUN apt install -y git
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

ENV PYTHON_VERSION=3.12.7
RUN pyenv install ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}


RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

WORKDIR /srv/app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY custom-nodes.yaml ./
COPY scripts/install_custom_nodes.py /usr/local/bin/install-custom-nodes
RUN python /usr/local/bin/install-custom-nodes --manifest custom-nodes.yaml --destination custom_nodes

COPY . .

CMD python main.py --listen 0.0.0.0
