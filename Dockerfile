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

# Model klasörlerini oluştur
RUN mkdir -p /app/models/checkpoints /app/models/loras /app/models/vae \
             /app/models/controlnet /app/models/upscale_models \
             /app/models/text_encoders /app/models/clip \
             /app/models/diffusion_models /app/models/unet \
             /app/models/embeddings /app/models/clip_vision

# Storage erişim testi - Build sırasında storage erişimi kontrol et
RUN echo "🔍 RunPod Build Storage Access Test" && \
    echo "=================================" && \
    echo "📁 Testing storage paths:" && \
    (ls -la /runpod-volume 2>/dev/null && echo "✅ /runpod-volume accessible" || echo "❌ /runpod-volume not accessible") && \
    (ls -la /workspace 2>/dev/null && echo "✅ /workspace accessible" || echo "❌ /workspace not accessible") && \
    (ls -la /content 2>/dev/null && echo "✅ /content accessible" || echo "❌ /content not accessible") && \
    echo "🌐 Network test:" && \
    (ping -c 1 google.com >/dev/null 2>&1 && echo "✅ Internet access available" || echo "❌ No internet access") && \
    echo "================================="

# Temel modelleri Docker build sırasında indir
RUN pip install huggingface-hub && \
    python download_models.py && \
    rm -rf /tmp/hf_cache

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
