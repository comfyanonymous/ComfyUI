# ComfyUI RunPod Serverless

Bu repository, ComfyUI'nin RunPod Serverless platformunda çalışması için optimize edilmiş versiyonudur.

## 🚀 Özellikler

- **RunPod Serverless** desteği
- **Otomatik scaling** ve queue yönetimi
- **GPU optimizasyonları** (FP16, FP8, XFormers)
- **Yeni API servisleri** (Minimax, ByteDance, Ideogram, vb.)
- **GitHub Actions CI/CD** pipeline
- **Otomatik build** tetikleme

## 📦 Kurulum

### RunPod Serverless Endpoint Oluşturma

1. **RunPod Dashboard**'a git
2. **Serverless** → **New Endpoint**
3. **GitHub Repository** seç:
   - Repository: `bahadirciloglu/ComfyUI`
   - Branch: `create_image`
   - Dockerfile Path: `Dockerfile`
   - Build Context: `.`

4. **GPU Configuration**:
   - Önerilen: 32GB veya 24GB
   - Worker Type: GPU
   - Endpoint Type: Queue

5. **Environment Variables** ekle:
   ```
   RUNPOD_API_KEY=your_api_key
   CIVITAI_API_KEY=your_civitai_key
   HUGGINGFACE_USERNAME=your_username
   HUGGINGFACE_PASSWORD=your_password
   COMFYUI_SERVERLESS=true
   COMFYUI_FAST_MODE=true
   ```

### GitHub Secrets Ayarlama

Repository Settings → Secrets and variables → Actions:

```
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id
```

## 🔄 CI/CD Pipeline

### Otomatik İş Akışı

1. **Code Push** → `create_image` branch
2. **GitHub Actions** çalışır:
   - Python syntax kontrolü
   - Dockerfile validasyonu
   - Docker build testi
   - Container registry'ye push
3. **Başarılı olursa** → RunPod build tetiklenir
4. **Hata varsa** → RunPod build tetiklenmez

### Manuel Test

```bash
# Sadece testleri çalıştır
gh workflow run test-only.yml

# Full pipeline çalıştır
git push origin create_image
```

## 🧪 Test Etme

### API Request Örneği

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "workflow": {
        "3": {
          "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0]
          },
          "class_type": "KSampler"
        },
        "4": {
          "inputs": {
            "ckpt_name": "sd_xl_base_1.0.safetensors"
          },
          "class_type": "CheckpointLoaderSimple"
        },
        "5": {
          "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 1
          },
          "class_type": "EmptyLatentImage"
        },
        "6": {
          "inputs": {
            "text": "a beautiful landscape",
            "clip": ["4", 1]
          },
          "class_type": "CLIPTextEncode"
        },
        "7": {
          "inputs": {
            "text": "blurry, low quality",
            "clip": ["4", 1]
          },
          "class_type": "CLIPTextEncode"
        },
        "8": {
          "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2]
          },
          "class_type": "VAEDecode"
        },
        "9": {
          "inputs": {
            "filename_prefix": "ComfyUI",
            "images": ["8", 0]
          },
          "class_type": "SaveImage"
        }
      }
    }
  }'
```

## 📊 Monitoring

### Build Status
- GitHub Actions: Repository → Actions tab
- RunPod: Dashboard → Serverless → Your Endpoint → Builds

### Logs
- RunPod Dashboard → Logs tab
- Real-time monitoring

### Metrics
- Request volume
- Worker scaling
- Billing information

## 🔧 Geliştirme

### Yerel Test

```bash
# Environment setup
cp .env.example .env
# .env dosyasını düzenle

# Docker build test
docker build -t comfyui-test .

# Container test
docker run -p 8000:8000 comfyui-test
```

### Yeni Özellik Ekleme

1. Feature branch oluştur
2. Değişiklikleri yap
3. Pull request aç
4. CI testleri geçince merge et
5. `create_image` branch'ına merge olunca otomatik deploy

## 📝 Environment Variables

Tüm environment variables için `.env.example` dosyasına bakın.

### Temel Ayarlar
- `RUNPOD_API_KEY`: RunPod API anahtarı
- `COMFYUI_SERVERLESS=true`: Serverless modu
- `COMFYUI_FAST_MODE=true`: Hızlı optimizasyonlar

### Performance Ayarları
- `COMFYUI_FP16_ACCUMULATION=true`: FP16 hızlandırma
- `COMFYUI_VRAM_MANAGEMENT=auto`: VRAM yönetimi
- `COMFYUI_CACHE_TYPE=none`: Serverless için cache

## 🆘 Sorun Giderme

### Build Hataları
1. GitHub Actions logs kontrol et
2. Dockerfile syntax kontrol et
3. Python syntax hataları düzelt

### Runtime Hataları
1. RunPod logs kontrol et
2. Environment variables kontrol et
3. Model dosyaları kontrol et

### Performance Sorunları
1. GPU memory kullanımı kontrol et
2. Batch size ayarla
3. Cache ayarlarını optimize et

## 📞 Destek

- GitHub Issues: Hata raporları ve özellik istekleri
- RunPod Discord: Platform desteği
- ComfyUI Community: Workflow yardımı