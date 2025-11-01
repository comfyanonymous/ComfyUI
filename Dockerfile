FROM python:3.13-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Tüm dosyaları kopyala
COPY . /app

# PyTorch CPU versiyonu ve bağımlılıklar
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# RunPod başlangıç scriptini çalıştırılabilir yap
RUN chmod +x start_runpod.py

# Environment variables
ENV PORT=8188
ENV RUNPOD_NETWORK_STORAGE_PATH=/runpod-volume
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV DO_NOT_TRACK=1

EXPOSE 8188

# RunPod başlangıç scriptini kullan
CMD ["python", "start_runpod.py"]
