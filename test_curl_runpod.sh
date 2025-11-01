#!/bin/bash
# RunPod Serverless API Test with curl
# Usage: ./test_curl_runpod.sh "your prompt here"

# Configuration
ENDPOINT_ID="sfkzjudvrj50yq"
API_KEY="${RUNPOD_API_KEY:-YOUR_API_KEY}"  # Set RUNPOD_API_KEY environment variable
BASE_URL="https://api.runpod.ai/v2"

# Check if API key is set
if [ "$API_KEY" = "YOUR_API_KEY" ]; then
    echo "❌ API key not set!"
    echo "💡 Set your API key: export RUNPOD_API_KEY='your_key_here'"
    echo "   Or edit this script and replace YOUR_API_KEY with your actual key"
    exit 1
fi

# Get prompt from command line or use default
PROMPT="${1:-modern software interface, clean dashboard design, professional UI layout, high quality, detailed}"

echo "🚀 Testing RunPod Serverless API..."
echo "📍 Endpoint: $ENDPOINT_ID"
echo "💬 Prompt: $PROMPT"

# Create the JSON payload
JSON_PAYLOAD=$(cat <<EOF
{
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
          "text": "$PROMPT",
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
          "filename_prefix": "curl_test",
          "images": ["8", 0]
        },
        "class_type": "SaveImage"
      }
    }
  }
}
EOF
)

# Submit the job
echo "📤 Submitting job..."
RESPONSE=$(curl -s -X POST "$BASE_URL/$ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "$JSON_PAYLOAD")

# Check if curl succeeded
if [ $? -ne 0 ]; then
    echo "❌ Curl command failed"
    exit 1
fi

# Parse job ID
JOB_ID=$(echo "$RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$JOB_ID" ]; then
    echo "❌ Failed to get job ID"
    echo "Response: $RESPONSE"
    exit 1
fi

echo "✅ Job submitted: $JOB_ID"

# Monitor job status
echo "⏳ Monitoring job status..."
MAX_WAIT=300  # 5 minutes
WAIT_TIME=0

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    STATUS_RESPONSE=$(curl -s -X GET "$BASE_URL/$ENDPOINT_ID/status/$JOB_ID" \
      -H "Authorization: Bearer $API_KEY")
    
    STATUS=$(echo "$STATUS_RESPONSE" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    
    case "$STATUS" in
        "COMPLETED")
            echo "✅ Job completed!"
            echo "📄 Full response:"
            echo "$STATUS_RESPONSE" | jq '.' 2>/dev/null || echo "$STATUS_RESPONSE"
            exit 0
            ;;
        "FAILED")
            echo "❌ Job failed!"
            echo "📄 Error response:"
            echo "$STATUS_RESPONSE" | jq '.' 2>/dev/null || echo "$STATUS_RESPONSE"
            exit 1
            ;;
        "IN_QUEUE"|"IN_PROGRESS")
            echo "⏳ Job status: $STATUS (${WAIT_TIME}s elapsed)"
            ;;
        *)
            echo "❓ Unknown status: $STATUS"
            ;;
    esac
    
    sleep 5
    WAIT_TIME=$((WAIT_TIME + 5))
done

echo "⏰ Timeout after ${MAX_WAIT} seconds"
echo "📄 Last status response:"
echo "$STATUS_RESPONSE" | jq '.' 2>/dev/null || echo "$STATUS_RESPONSE"