FROM python:3.12-slim
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=on
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install git -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio --extra-index-url https://download.pytorch.org/whl/cu124 \
    && pip install -r requirements.txt

COPY . .

CMD ["python3", "main.py"]