#!/usr/bin/env python3
"""
RunPod Serverless API Test - pars klasöründeki resimlere benzer resim üretimi
"""

import os
import json
import base64
import requests
import time
from pathlib import Path

class RunPodTester:
    def __init__(self):
        # RunPod API bilgileri (.env.example'dan alın)
        self.api_key = os.getenv("RUNPOD_API_KEY", "YOUR_RUNPOD_API_KEY")  # .env dosyasından alın
        self.endpoint_id = "sfkzjudvrj50yq"  # Dashboard'dan aldığınız ID
        self.base_url = "https://api.runpod.ai/v2"
        
    def create_test_workflow(self, prompt):
        """Test workflow'u oluştur"""
        return {
            "input": {
                "workflow": {
                    "3": {
                        "inputs": {
                            "seed": 42,
                            "steps": 25,
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
                            "text": prompt,
                            "clip": ["4", 1]
                        },
                        "class_type": "CLIPTextEncode"
                    },
                    "7": {
                        "inputs": {
                            "text": "blurry, low quality, distorted, ugly, bad anatomy, worst quality",
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
                            "filename_prefix": "runpod_test",
                            "images": ["8", 0]
                        },
                        "class_type": "SaveImage"
                    }
                }
            }
        }
    
    def submit_job(self, prompt):
        """RunPod'a job gönder"""
        try:
            workflow_data = self.create_test_workflow(prompt)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            url = f"{self.base_url}/{self.endpoint_id}/run"
            
            print(f"🚀 Job gönderiliyor...")
            print(f"   Endpoint: {self.endpoint_id}")
            print(f"   Prompt: {prompt[:50]}...")
            
            response = requests.post(url, json=workflow_data, headers=headers, timeout=30)
            response.raise_for_status()
            
            job_data = response.json()
            job_id = job_data.get("id")
            
            print(f"✅ Job gönderildi: {job_id}")
            return job_id
            
        except Exception as e:
            print(f"❌ Job gönderme hatası: {e}")
            return None
    
    def check_job_status(self, job_id):
        """Job durumunu kontrol et"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            url = f"{self.base_url}/{self.endpoint_id}/status/{job_id}"
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"❌ Durum kontrolü hatası: {e}")
            return None
    
    def wait_for_completion(self, job_id, timeout=300):
        """Job'ın tamamlanmasını bekle"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status_data = self.check_job_status(job_id)
            
            if not status_data:
                time.sleep(5)
                continue
                
            status = status_data.get("status")
            
            if status == "COMPLETED":
                print("✅ Job tamamlandı!")
                return status_data.get("output")
            elif status == "FAILED":
                print("❌ Job başarısız!")
                print(f"   Hata: {status_data.get('error', 'Bilinmeyen hata')}")
                return None
            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                elapsed = int(time.time() - start_time)
                print(f"⏳ Job çalışıyor... ({status}) - {elapsed}s")
            
            time.sleep(5)
        
        print(f"⏰ Timeout: {timeout} saniye")
        return None
    
    def save_results(self, output_data, test_name):
        """Sonuçları kaydet"""
        if not output_data:
            return []
        
        os.makedirs("test_output", exist_ok=True)
        saved_files = []
        
        outputs = output_data.get("outputs", [])
        for i, output in enumerate(outputs):
            if "data" in output:
                try:
                    # Base64 decode
                    image_data = base64.b64decode(output["data"])
                    
                    # Dosya adı
                    filename = f"{test_name}_{i+1:02d}.png"
                    filepath = f"test_output/{filename}"
                    
                    # Kaydet
                    with open(filepath, "wb") as f:
                        f.write(image_data)
                    
                    saved_files.append(filepath)
                    print(f"💾 Resim kaydedildi: {filepath}")
                    
                except Exception as e:
                    print(f"❌ Resim kaydetme hatası: {e}")
        
        return saved_files

def main():
    """Ana test fonksiyonu"""
    print("🚀 RunPod Serverless Test Başlatılıyor...")
    
    tester = RunPodTester()
    
    # API key kontrolü
    if tester.api_key == "YOUR_RUNPOD_API_KEY":
        print("❌ API key ayarlanmamış!")
        print("💡 test_runpod_api.py dosyasında API key'inizi güncelleyin")
        print("   API key'i RunPod dashboard → Settings → API Keys'den alabilirsiniz")
        return
    
    # pars klasöründeki screenshot'lara benzer prompt'lar
    test_prompts = [
        "modern software interface, clean dashboard design, professional UI layout, high quality, detailed",
        "web application screenshot, user interface mockup, modern design system, clean layout",
        "software dashboard, data visualization interface, professional web app design, modern UI",
        "application interface design, clean user experience, professional software layout, modern style",
        "digital interface mockup, clean web design, professional dashboard layout, modern UI elements"
    ]
    
    print(f"\n🎨 {len(test_prompts)} adet test resmi üretiliyor...")
    print(f"📍 Endpoint: {tester.endpoint_id}")
    
    all_results = []
    
    # Her prompt için test
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        
        # Job gönder
        job_id = tester.submit_job(prompt)
        if not job_id:
            continue
        
        # Tamamlanmasını bekle
        output = tester.wait_for_completion(job_id)
        
        # Sonuçları kaydet
        if output:
            saved_files = tester.save_results(output, f"ui_design_test_{i}")
            all_results.extend(saved_files)
            print(f"✅ Test {i} başarılı: {len(saved_files)} resim")
        else:
            print(f"❌ Test {i} başarısız")
        
        # Rate limiting için bekleme
        if i < len(test_prompts):
            print("⏸️  Rate limiting için 10 saniye bekleniyor...")
            time.sleep(10)
    
    # Özet
    print(f"\n🎉 Test tamamlandı!")
    print(f"📊 Toplam üretilen resim: {len(all_results)}")
    print(f"📁 Kaydedilen dosyalar:")
    for file in all_results:
        print(f"   - {file}")
    
    if all_results:
        print(f"\n💡 Resimleri görüntülemek için:")
        print(f"   open test_output/")

if __name__ == "__main__":
    main()