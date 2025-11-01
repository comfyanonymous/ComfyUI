#!/bin/bash

# RunPod Build Trigger Script
# Bu script GitHub Actions'dan RunPod build'ini tetikler

set -e

RUNPOD_API_KEY="$1"
ENDPOINT_ID="$2"
GITHUB_SHA="$3"
GITHUB_REF="$4"

if [ -z "$RUNPOD_API_KEY" ] || [ -z "$ENDPOINT_ID" ]; then
    echo "❌ Error: RUNPOD_API_KEY and ENDPOINT_ID are required"
    exit 1
fi

echo "🚀 Triggering RunPod build for endpoint: $ENDPOINT_ID"
echo "📝 Git SHA: $GITHUB_SHA"
echo "🌿 Git Ref: $GITHUB_REF"

# RunPod API endpoint
API_URL="https://api.runpod.ai/v2/$ENDPOINT_ID"

# Check endpoint status first
echo "🔍 Checking endpoint status..."
STATUS_RESPONSE=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" "$API_URL")

if echo "$STATUS_RESPONSE" | grep -q "error"; then
    echo "❌ Error checking endpoint status:"
    echo "$STATUS_RESPONSE"
    exit 1
fi

echo "✅ Endpoint is accessible"

# Trigger rebuild
echo "🔄 Triggering rebuild..."
REBUILD_RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
        \"action\": \"rebuild\",
        \"metadata\": {
            \"github_sha\": \"$GITHUB_SHA\",
            \"github_ref\": \"$GITHUB_REF\",
            \"triggered_by\": \"github_actions\",
            \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
        }
    }" \
    "$API_URL/rebuild")

if echo "$REBUILD_RESPONSE" | grep -q "error"; then
    echo "❌ Error triggering rebuild:"
    echo "$REBUILD_RESPONSE"
    exit 1
fi

echo "✅ RunPod rebuild triggered successfully!"
echo "📊 Response: $REBUILD_RESPONSE"

# Wait a bit and check build status
echo "⏳ Waiting 10 seconds before checking build status..."
sleep 10

BUILD_STATUS=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" "$API_URL/builds")
echo "🏗️ Build Status: $BUILD_STATUS"

echo "🎉 RunPod deployment pipeline completed!"
echo ""
echo "📋 Next Steps:"
echo "  1. Monitor build progress in RunPod dashboard"
echo "  2. Test the endpoint once build completes"
echo "  3. Check logs if any issues occur"