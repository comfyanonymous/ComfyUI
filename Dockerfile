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

# Cloud Run PORT environment variable'ını kullan
ENV PORT=8188
EXPOSE 8188

# Başlat
CMD python main.py --listen 0.0.0.0 --port ${PORT}
