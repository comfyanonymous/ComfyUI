#!/usr/bin/env python3
"""
Docker build sırasında temel modelleri indir
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_model(repo_id, filename, target_dir):
    """Model indir ve hedef dizine kopyala"""
    try:
        print(f"📥 İndiriliyor: {repo_id}/{filename}")
        
        # Model'i indir
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir="/tmp/hf_cache"
        )
        
        # Hedef dizini oluştur
        os.makedirs(target_dir, exist_ok=True)
        
        # Dosyayı kopyala
        target_path = os.path.join(target_dir, filename)
        os.system(f"cp '{model_path}' '{target_path}'")
        
        print(f"✅ Kaydedildi: {target_path}")
        return True
        
    except Exception as e:
        print(f"❌ Hata: {repo_id}/{filename} - {e}")
        return False

def main():
    """Temel modelleri indir"""
    print("🚀 Docker build - Model indirme başlatılıyor...")
    
    models_base = "/app/models"
    
    # İndirilecek modeller
    models_to_download = [
        # SDXL Base Model
        {
            "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "filename": "sd_xl_base_1.0.safetensors",
            "target_dir": f"{models_base}/checkpoints"
        },
        # SDXL VAE
        {
            "repo_id": "stabilityai/sdxl-vae", 
            "filename": "sdxl_vae.safetensors",
            "target_dir": f"{models_base}/vae"
        },
        # CLIP Text Encoder
        {
            "repo_id": "openai/clip-vit-large-patch14",
            "filename": "pytorch_model.bin",
            "target_dir": f"{models_base}/clip"
        }
    ]
    
    success_count = 0
    
    for model in models_to_download:
        if download_model(
            model["repo_id"], 
            model["filename"], 
            model["target_dir"]
        ):
            success_count += 1
    
    print(f"\n🎉 Model indirme tamamlandı: {success_count}/{len(models_to_download)}")
    
    # Model klasörlerini listele
    print("\n📁 Model klasörleri:")
    for root, dirs, files in os.walk(models_base):
        level = root.replace(models_base, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_size = os.path.getsize(os.path.join(root, file))
            size_mb = file_size / (1024 * 1024)
            print(f"{subindent}{file} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()