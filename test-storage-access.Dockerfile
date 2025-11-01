FROM python:3.10-slim

# RunPod build sırasında storage erişimi test et
RUN echo "🔍 RunPod Build Storage Access Test" && \
    echo "=================================" && \
    echo "" && \
    echo "📁 Testing common RunPod storage paths:" && \
    echo "" && \
    echo "1. /workspace:" && \
    (ls -la /workspace 2>/dev/null || echo "❌ /workspace not accessible") && \
    echo "" && \
    echo "2. /runpod-volume:" && \
    (ls -la /runpod-volume 2>/dev/null || echo "❌ /runpod-volume not accessible") && \
    echo "" && \
    echo "3. /content:" && \
    (ls -la /content 2>/dev/null || echo "❌ /content not accessible") && \
    echo "" && \
    echo "4. /storage:" && \
    (ls -la /storage 2>/dev/null || echo "❌ /storage not accessible") && \
    echo "" && \
    echo "5. Environment variables:" && \
    env | grep -i runpod || echo "❌ No RUNPOD env vars found" && \
    echo "" && \
    echo "6. Mount points:" && \
    mount | grep -E "(workspace|runpod|storage)" || echo "❌ No storage mounts found" && \
    echo "" && \
    echo "7. Available disk space:" && \
    df -h && \
    echo "" && \
    echo "8. Network connectivity test:" && \
    (ping -c 1 google.com >/dev/null 2>&1 && echo "✅ Internet access available" || echo "❌ No internet access") && \
    echo "" && \
    echo "=================================" && \
    echo "🏁 Test completed"

# Basit bir uygulama
WORKDIR /app
RUN echo 'print("Hello from RunPod build test!")' > app.py

CMD ["python", "app.py"]