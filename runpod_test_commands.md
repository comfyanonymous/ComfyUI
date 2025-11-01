# RunPod Container Test Komutları

## 1. Container'a Bağlan
RunPod dashboard'da "Connect" → "Start Web Terminal"

## 2. Test Scriptini Çalıştır
```bash
# Container içinde:
cd /app
python test_image_generation.py
```

## 3. Manuel Test (Alternatif)
```bash
# ComfyUI durumunu kontrol et
curl http://127.0.0.1:8188/system_stats

# Model listesini al
curl http://127.0.0.1:8188/object_info | jq '.CheckpointLoaderSimple.input.required.ckpt_name'

# Basit workflow gönder
curl -X POST http://127.0.0.1:8188/prompt \
  -H "Content-Type: application/json" \
  -d @test_simple_workflow.json
```

## 4. Network Storage Kontrolü
```bash
# Models klasörünü kontrol et
ls -la /app/models/
ls -la /runpod-volume/models/

# Symlink kontrolü
readlink /app/models
```

## 5. Model Yükleme (Gerekirse)
```bash
# Network storage'a model yükle
cd /runpod-volume/models/checkpoints
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# ComfyUI'yi yeniden başlat
pkill -f main.py
python main.py --listen 0.0.0.0 --port 8188 --cpu
```

## 6. Beklenen Çıktılar

### Başarılı Durum:
```
✅ ComfyUI server çalışıyor
✅ Yüklü modeller (1 adet):
  - sd_xl_base_1.0.safetensors
🎨 Resim üretiliyor...
   Model: sd_xl_base_1.0.safetensors
   Prompt: a modern user interface design...
📝 Prompt ID: abc123
⏳ Bekleniyor... (15s)
✅ Resim kaydedildi: test_output/test_ui_design_00001_.png
🎉 Test tamamlandı!
```

### Hata Durumu:
```
❌ ComfyUI server'a bağlanılamıyor
❌ Hiç model bulunamadı
❌ Network storage mount edilmedi
```

## 7. Troubleshooting

### ComfyUI Başlamazsa:
```bash
# Logları kontrol et
tail -f /var/log/comfyui.log

# Manuel başlat
cd /app
python main.py --listen 0.0.0.0 --port 8188 --cpu
```

### Network Storage Sorunları:
```bash
# Mount durumunu kontrol et
mount | grep runpod-volume

# Manuel mount
mkdir -p /runpod-volume/models
ln -sf /runpod-volume/models /app/models
```