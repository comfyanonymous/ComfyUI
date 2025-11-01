#!/usr/bin/env python3
"""
ComfyUI Test Script - pars klasöründeki resimlere benzer resim üretimi
"""

import os
import json
import base64
import requests
from pathlib import Path
import time

class ComfyUITester:
    def __init__(self, comfyui_url="http://127.0.0.1:8188"):
        self.comfyui_url = comfyui_url
        self.client_id = "test_client"
        
    def check_server(self):
        """ComfyUI server'ın çalışıp çalışmadığını kontrol et"""
        try:
            response = requests.get(f"{self.comfyui_url}/system_stats", timeout=5)
            if response.status_code == 200:
                print("✅ ComfyUI server çalışıyor")
                return True
            else:
                print(f"❌ ComfyUI server yanıt vermiyor: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ ComfyUI server'a bağlanılamıyor: {e}")
            return False
    
    def get_models(self):
        """Yüklü modelleri listele"""
        try:
            response = requests.get(f"{self.comfyui_url}/object_info")
            if response.status_code == 200:
                data = response.json()
                checkpoints = data.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [])
                if isinstance(checkpoints, list) and len(checkpoints) > 1:
                    models = checkpoints[0]  # İlk element model listesi
                    print(f"✅ Yüklü modeller ({len(models)} adet):")
                    for model in models[:5]:  # İlk 5 modeli göster
                        print(f"  - {model}")
                    return models
                else:
                    print("❌ Model listesi alınamadı")
                    return []
        except Exception as e:
            print(f"❌ Model listesi alınamadı: {e}")
            return []
    
    def create_basic_workflow(self, model_name="sd_xl_base_1.0.safetensors", prompt="a beautiful landscape"):
        """Temel SDXL workflow oluştur"""
        workflow = {
            "3": {
                "inputs": {
                    "seed": 42,
                    "steps": 20,
                    "cfg": 8.0,
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
                    "ckpt_name": model_name
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
                    "text": prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": "blurry, low quality, distorted",
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
                    "filename_prefix": "test_output",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }
        return workflow
    
    def generate_image(self, prompt, model_name=None):
        """Resim üret"""
        try:
            # Model seç
            if not model_name:
                models = self.get_models()
                if not models:
                    print("❌ Hiç model bulunamadı")
                    return None
                model_name = models[0]  # İlk modeli kullan
            
            print(f"🎨 Resim üretiliyor...")
            print(f"   Model: {model_name}")
            print(f"   Prompt: {prompt}")
            
            # Workflow oluştur
            workflow = self.create_basic_workflow(model_name, prompt)
            
            # İsteği gönder
            response = requests.post(
                f"{self.comfyui_url}/prompt",
                json={
                    "prompt": workflow,
                    "client_id": self.client_id
                },
                timeout=30
            )
            response.raise_for_status()
            
            prompt_id = response.json()["prompt_id"]
            print(f"📝 Prompt ID: {prompt_id}")
            
            # Tamamlanmasını bekle
            return self.wait_for_completion(prompt_id)
            
        except Exception as e:
            print(f"❌ Resim üretimi başarısız: {e}")
            return None
    
    def wait_for_completion(self, prompt_id, timeout=300):
        """İşlemin tamamlanmasını bekle"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Queue durumunu kontrol et
                queue_response = requests.get(f"{self.comfyui_url}/queue")
                queue_data = queue_response.json()
                
                # İşlem hala çalışıyor mu?
                running = any(item[1]["prompt_id"] == prompt_id for item in queue_data.get("queue_running", []))
                pending = any(item[1]["prompt_id"] == prompt_id for item in queue_data.get("queue_pending", []))
                
                if not running and not pending:
                    # İşlem tamamlandı, sonuçları al
                    history_response = requests.get(f"{self.comfyui_url}/history/{prompt_id}")
                    if history_response.status_code == 200:
                        history_data = history_response.json()
                        if prompt_id in history_data:
                            return self.download_results(history_data[prompt_id])
                
                print(f"⏳ Bekleniyor... ({int(time.time() - start_time)}s)")
                time.sleep(3)
                
            except Exception as e:
                print(f"❌ Durum kontrolü hatası: {e}")
                time.sleep(5)
        
        print(f"⏰ Timeout: {timeout} saniye")
        return None
    
    def download_results(self, history_data):
        """Sonuçları indir"""
        results = []
        
        if "outputs" in history_data:
            for node_id, node_output in history_data["outputs"].items():
                if "images" in node_output:
                    for image_info in node_output["images"]:
                        try:
                            # Resmi indir
                            image_url = f"{self.comfyui_url}/view"
                            params = {
                                "filename": image_info["filename"],
                                "subfolder": image_info.get("subfolder", ""),
                                "type": image_info.get("type", "output")
                            }
                            
                            image_response = requests.get(image_url, params=params)
                            image_response.raise_for_status()
                            
                            # Test output klasörüne kaydet
                            os.makedirs("test_output", exist_ok=True)
                            output_path = f"test_output/{image_info['filename']}"
                            
                            with open(output_path, "wb") as f:
                                f.write(image_response.content)
                            
                            results.append({
                                "filename": image_info["filename"],
                                "path": output_path,
                                "node_id": node_id
                            })
                            
                            print(f"✅ Resim kaydedildi: {output_path}")
                            
                        except Exception as e:
                            print(f"❌ Resim indirme hatası: {e}")
        
        return results

def main():
    """Ana test fonksiyonu"""
    print("🚀 ComfyUI Test Başlatılıyor...")
    
    tester = ComfyUITester()
    
    # Server kontrolü
    if not tester.check_server():
        print("\n💡 ComfyUI'yi başlatmak için:")
        print("   python main.py --listen 0.0.0.0 --port 8188")
        return
    
    # Model kontrolü
    models = tester.get_models()
    if not models:
        print("\n💡 Model yüklemek için:")
        print("   1. models/checkpoints/ klasörüne .safetensors dosyası ekleyin")
        print("   2. ComfyUI'yi yeniden başlatın")
        return
    
    # Test prompt'ları (pars klasöründeki screenshot'lara benzer)
    test_prompts = [
        "a modern user interface design, clean layout, professional software interface",
        "screenshot of a web application, dashboard design, modern UI elements",
        "software interface mockup, clean design, professional layout",
        "application screenshot, user interface design, modern web app",
        "dashboard interface, clean UI design, professional software layout"
    ]
    
    print(f"\n🎨 {len(test_prompts)} adet test resmi üretiliyor...")
    
    # Her prompt için resim üret
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        result = tester.generate_image(prompt, models[0])
        
        if result:
            print(f"✅ Test {i} başarılı: {len(result)} resim üretildi")
        else:
            print(f"❌ Test {i} başarısız")
        
        # Kısa bekleme
        time.sleep(2)
    
    print("\n🎉 Test tamamlandı!")
    print("📁 Üretilen resimler: test_output/ klasöründe")

if __name__ == "__main__":
    main()