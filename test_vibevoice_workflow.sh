#!/bin/bash

# Test ComfyUI API with VibeVoice workflow
# Usage: ./test_vibevoice_workflow.sh [API_KEY]

# Configuration
BASE_URL="http://localhost:8188"
API_KEY="${1:-}"

# Set headers based on whether API key is provided
if [ -n "$API_KEY" ]; then
    AUTH_HEADER="Authorization: Bearer $API_KEY"
    echo "Using API Key authentication"
else
    AUTH_HEADER=""
    echo "No API Key provided (running without authentication)"
fi

# The workflow payload
# This converts the ComfyUI workflow format to the prompt API format
read -r -d '' PAYLOAD << 'EOF'
{
  "prompt": {
    "1": {
      "inputs": {
        "speaker_1_voice": ["2", 0],
        "speaker_2_voice": null,
        "speaker_3_voice": null,
        "speaker_4_voice": null,
        "model_name": "VibeVoice-Large",
        "text": "[1] And this is a generated voice, how cool is that?",
        "quantize_llm_4bit": false,
        "attention_mode": "sdpa",
        "cfg_scale": 1.3,
        "inference_steps": 10,
        "seed": 1117544514407045,
        "do_sample": true,
        "temperature": 0.95,
        "top_p": 0.95,
        "top_k": 0,
        "force_offload": false
      },
      "class_type": "VibeVoiceTTS"
    },
    "2": {
      "inputs": {
        "audio": "audio1.wav"
      },
      "class_type": "LoadAudio"
    },
    "3": {
      "inputs": {
        "audio": ["1", 0],
        "filename_prefix": "audio/ComfyUI"
      },
      "class_type": "SaveAudio"
    }
  },
  "client_id": "test_client_$(date +%s)"
}
EOF

echo ""
echo "================================================"
echo "Sending workflow to ComfyUI..."
echo "================================================"
echo ""

# Make the request
if [ -n "$AUTH_HEADER" ]; then
    response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -H "$AUTH_HEADER" \
        -d "$PAYLOAD" \
        "$BASE_URL/prompt")
else
    response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD" \
        "$BASE_URL/prompt")
fi

# Extract HTTP status
http_status=$(echo "$response" | grep "HTTP_STATUS" | cut -d':' -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

echo "HTTP Status: $http_status"
echo ""
echo "Response:"
echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
echo ""

if [ "$http_status" = "200" ]; then
    echo "✓ Workflow queued successfully!"
    
    # Extract prompt_id if available
    prompt_id=$(echo "$body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('prompt_id', ''))" 2>/dev/null)
    if [ -n "$prompt_id" ]; then
        echo "Prompt ID: $prompt_id"
        echo ""
        echo "To check status:"
        if [ -n "$AUTH_HEADER" ]; then
            echo "  curl -H \"$AUTH_HEADER\" $BASE_URL/history/$prompt_id"
        else
            echo "  curl $BASE_URL/history/$prompt_id"
        fi
    fi
elif [ "$http_status" = "401" ]; then
    echo "✗ Authentication failed - check your API key"
else
    echo "✗ Request failed"
fi

echo ""
echo "================================================"
