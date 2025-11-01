#!/bin/bash
# Debug RunPod API call

ENDPOINT_ID="sfkzjudvrj50yq"
API_KEY="${RUNPOD_API_KEY:-YOUR_API_KEY}"
BASE_URL="https://api.runpod.ai/v2"

echo "🔍 Debugging RunPod API..."
echo "📍 Endpoint: $ENDPOINT_ID"
echo "🔑 API Key: ${API_KEY:0:10}..." # Show only first 10 chars for security

# Simple test with minimal payload
echo "📤 Testing with simple payload..."
curl -v -X POST "$BASE_URL/$ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"input":{"prompt":"test prompt"}}'