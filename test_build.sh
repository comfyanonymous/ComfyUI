#!/bin/bash
# RunPod Build Test Script

echo "🔍 RunPod Storage Access Test"
echo "============================="

# Docker build ile test et (local)
echo "📦 Local Docker build test..."
docker build -f test-storage-access.Dockerfile -t runpod-storage-test . 2>&1 | grep -E "(Storage Access Test|accessible|not accessible)"

echo ""
echo "🚀 Gerçek test için RunPod Dashboard'da bu Dockerfile'ı kullan:"
echo "   1. RunPod Dashboard → Templates → Create Template"
echo "   2. Container Image → Build from Dockerfile"
echo "   3. test-storage-access.Dockerfile içeriğini yapıştır"
echo "   4. Build loglarında storage erişim sonuçlarını kontrol et"

echo ""
echo "📋 Test sonuçlarına göre:"
echo "   ✅ Storage erişimi VARSA → Build sırasında model kopyalama"
echo "   ❌ Storage erişimi YOKSA → HuggingFace'den model indirme"