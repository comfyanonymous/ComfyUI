# RunPod Network Storage Setup Guide

## 1. Network Storage Hazırlığı

### RunPod Dashboard'da:
1. **Network Storage** oluşturun (örn: `comfyui-models`)
2. Storage boyutunu belirleyin (en az 50GB önerilir)
3. Storage ID'sini not alın

### Modelleri Network Storage'a Yükleme:
```bash
# RunPod pod'unda terminal açın
cd /runpod-volume

# Models klasörü oluşturun
mkdir -p models/{checkpoints,loras,vae,controlnet,upscale_models,text_encoders,clip,diffusion_models,unet,embeddings,clip_vision}

# Örnek model indirme (SDXL Base)
cd models/checkpoints
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# VAE modeli
cd ../vae
wget https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors

# ControlNet modeli
cd ../controlnet
wget https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_mid.safetensors
```

## 2. RunPod Template Ayarları

### Container Settings:
- **Docker Image**: `your-registry/comfyui-runpod:latest`
- **Container Disk**: 20GB (minimum)
- **Network Storage**: Mount ettiğiniz storage'ı seçin

### Environment Variables:
```bash
# Zorunlu
RUNPOD_NETWORK_STORAGE_PATH=/runpod-volume
PORT=8188

# Opsiyonel
LISTEN=0.0.0.0
COMFYUI_ARGS=--preview-method auto
DOWNLOAD_MODELS=sdxl-base,sdxl-vae  # Otomatik indirme için
```

### Ports:
- **Container Port**: 8188
- **Expose HTTP Ports**: 8188

## 3. Model Klasör Yapısı

Network storage'da şu yapı oluşturulmalı:
```
/runpod-volume/
└── models/
    ├── checkpoints/          # Ana modeller (SDXL, SD 1.5, vb.)
    │   ├── sd_xl_base_1.0.safetensors
    │   └── sd_xl_refiner_1.0.safetensors
    ├── loras/               # LoRA modelleri
    ├── vae/                 # VAE modelleri
    │   └── sdxl_vae.safetensors
    ├── controlnet/          # ControlNet modelleri
    ├── upscale_models/      # Upscaler modeller
    ├── text_encoders/       # CLIP modelleri
    ├── clip/               # Legacy CLIP klasörü
    ├── diffusion_models/    # UNet modelleri
    ├── unet/               # Legacy UNet klasörü
    ├── embeddings/         # Textual Inversion
    └── clip_vision/        # CLIP Vision modelleri
```

## 4. Popüler Modeller

### Checkpoints:
```bash
# SDXL Base
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# SDXL Refiner
wget https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors

# SD 1.5
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
```

### VAE:
```bash
# SDXL VAE
wget https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors

# SD 1.5 VAE
wget https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors
```

### ControlNet:
```bash
# SDXL Canny
wget https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_mid.safetensors

# SDXL Depth
wget https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_depth_mid.safetensors
```

## 5. Deployment

### Build ve Push:
```bash
# Docker image build
docker build -t your-registry/comfyui-runpod:latest .

# Registry'ye push
docker push your-registry/comfyui-runpod:latest
```

### RunPod'da Deploy:
1. Template oluşturun
2. Network storage'ı mount edin
3. Environment variables'ları ayarlayın
4. Deploy edin

## 6. Monitoring

Container loglarında şunları göreceksiniz:
```
2024-11-01 13:15:00 - INFO - RunPod ComfyUI başlatılıyor...
2024-11-01 13:15:01 - INFO - Network storage hazır
2024-11-01 13:15:02 - INFO - Models klasörü network storage'a bağlandı
2024-11-01 13:15:03 - INFO - Model klasörü: checkpoints (2 dosya)
2024-11-01 13:15:04 - INFO - ComfyUI başlatılıyor...
```

## 7. Troubleshooting

### Network Storage Mount Edilmezse:
- Local models klasörü kullanılır
- Logları kontrol edin: `RUNPOD_NETWORK_STORAGE_PATH` doğru mu?

### Modeller Bulunamazsa:
- Network storage'da model dosyaları var mı kontrol edin
- Dosya izinlerini kontrol edin
- Symlink'in doğru çalıştığını kontrol edin

### Performance İyileştirme:
- GPU instance kullanın (CPU yerine)
- Dockerfile'da `--cpu` parametresini kaldırın
- CUDA support ekleyin