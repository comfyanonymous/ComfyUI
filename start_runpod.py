#!/usr/bin/env python3
"""
RunPod başlangıç scripti - Network storage mount ve model yönetimi
"""

import os
import sys
import logging
import subprocess
import time

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def mount_runpod_storage():
    """RunPod network storage'ı mount et"""
    try:
        # RunPod network storage path (environment variable'dan al)
        network_storage_path = os.environ.get('RUNPOD_NETWORK_STORAGE_PATH', '/runpod-volume')
        models_storage_path = os.path.join(network_storage_path, 'models')

        # Local models klasörü
        local_models_path = '/app/models'

        logger.info(f"Network storage path: {network_storage_path}")
        logger.info(f"Models storage path: {models_storage_path}")

        # Network storage'da models klasörü var mı kontrol et
        if os.path.exists(models_storage_path):
            logger.info("Network storage'da models klasörü bulundu")

            # Local models klasörünü sil ve symlink oluştur
            if os.path.exists(local_models_path):
                if os.path.islink(local_models_path):
                    os.unlink(local_models_path)
                else:
                    import shutil
                    shutil.rmtree(local_models_path)

            # Symlink oluştur
            os.symlink(models_storage_path, local_models_path)
            logger.info(f"Models klasörü network storage'a bağlandı: {models_storage_path} -> {local_models_path}")

            # Model klasörlerini kontrol et
            check_model_folders(local_models_path)

        else:
            logger.warning(f"Network storage'da models klasörü bulunamadı: {models_storage_path}")
            logger.info("Local models klasörü kullanılacak")

            # Network storage'da models klasörü oluştur
            os.makedirs(models_storage_path, exist_ok=True)
            logger.info(f"Network storage'da models klasörü oluşturuldu: {models_storage_path}")

            # Mevcut local models'i network storage'a taşı
            if os.path.exists(local_models_path) and not os.path.islink(local_models_path):
                import shutil
                shutil.copytree(local_models_path, models_storage_path, dirs_exist_ok=True)
                shutil.rmtree(local_models_path)
                logger.info("Local models network storage'a taşındı")

            # Symlink oluştur
            os.symlink(models_storage_path, local_models_path)
            logger.info("Models klasörü network storage'a bağlandı")

    except Exception as e:
        logger.error(f"Network storage mount hatası: {e}")
        logger.info("Local models klasörü kullanılacak")
        ensure_local_model_folders()

def check_model_folders(models_path):
    """Model klasörlerinin varlığını kontrol et"""
    required_folders = [
        'checkpoints',
        'loras',
        'vae',
        'controlnet',
        'upscale_models',
        'text_encoders',
        'clip',
        'diffusion_models',
        'unet',
        'embeddings',
        'clip_vision'
    ]

    for folder in required_folders:
        folder_path = os.path.join(models_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"Model klasörü oluşturuldu: {folder}")
        else:
            # Klasördeki dosya sayısını kontrol et
            file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            logger.info(f"Model klasörü: {folder} ({file_count} dosya)")

def ensure_local_model_folders():
    """Local model klasörlerini oluştur"""
    models_path = '/app/models'
    check_model_folders(models_path)

def download_essential_models():
    """Temel modelleri indir (opsiyonel)"""
    try:
        models_to_download = os.environ.get('DOWNLOAD_MODELS', '').split(',')
        models_to_download = [m.strip() for m in models_to_download if m.strip()]

        if not models_to_download:
            logger.info("İndirilecek model belirtilmedi")
            return

        logger.info(f"İndirilecek modeller: {models_to_download}")

        # Burada model indirme logic'i eklenebilir
        # Örnek: huggingface-hub kullanarak

    except Exception as e:
        logger.error(f"Model indirme hatası: {e}")

def setup_environment():
    """Çevre değişkenlerini ayarla"""
    # ComfyUI için gerekli environment variables
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'

    # RunPod specific
    if 'RUNPOD_POD_ID' in os.environ:
        logger.info(f"RunPod Pod ID: {os.environ['RUNPOD_POD_ID']}")

    # Port ayarı
    port = os.environ.get('PORT', '8188')
    os.environ['PORT'] = port
    logger.info(f"Server port: {port}")

def wait_for_storage():
    """Network storage'ın hazır olmasını bekle"""
    max_wait = 30  # 30 saniye
    wait_interval = 2

    network_storage_path = os.environ.get('RUNPOD_NETWORK_STORAGE_PATH', '/runpod-volume')

    for i in range(0, max_wait, wait_interval):
        if os.path.exists(network_storage_path):
            logger.info("Network storage hazır")
            return True

        logger.info(f"Network storage bekleniyor... ({i}/{max_wait}s)")
        time.sleep(wait_interval)

    logger.warning("Network storage timeout - local storage kullanılacak")
    return False

def main():
    """Ana başlangıç fonksiyonu"""
    logger.info("RunPod ComfyUI başlatılıyor...")

    # Environment setup
    setup_environment()

    # Network storage'ı bekle
    wait_for_storage()

    # Network storage mount
    mount_runpod_storage()

    # Temel modelleri indir (opsiyonel)
    download_essential_models()

    # ComfyUI'yi başlat
    logger.info("ComfyUI başlatılıyor...")

    # Port ve listen address
    port = os.environ.get('PORT', '8188')
    listen = os.environ.get('LISTEN', '0.0.0.0')

    # ComfyUI command - main.py dosyası mevcut dizinde
    cmd = [
        sys.executable, 'main.py',
        '--listen', listen,
        '--port', port,
        '--cpu'  # CPU mode for RunPod
    ]

    # Extra args
    if os.environ.get('COMFYUI_ARGS'):
        extra_args = os.environ['COMFYUI_ARGS'].split()
        cmd.extend(extra_args)

    logger.info(f"ComfyUI komutu: {' '.join(cmd)}")

    # ComfyUI'yi başlat
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("ComfyUI durduruldu")
    except Exception as e:
        logger.error(f"ComfyUI başlatma hatası: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
