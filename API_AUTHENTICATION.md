# API Key Authentication and Health Check

## Overview

This implementation adds API key authentication protection to the ComfyUI REST API and a health check endpoint.

## Features

### 1. API Key Authentication

Protects all API endpoints (except exempt ones) with API key authentication.

#### Configuration

You can enable API key authentication in two ways:

**Option 1: Command line argument**
```bash
python main.py --api-key "your-secret-api-key-here"
```

**Option 2: API key file (more secure)**
```bash
# Create a file with your API key
echo "your-secret-api-key-here" > api_key.txt

# Start ComfyUI with the API key file
python main.py --api-key-file api_key.txt
```

#### Using the API with Authentication

When API key authentication is enabled, you must provide the API key in your requests:

**Method 1: Authorization Header (Bearer Token)**
```bash
curl -H "Authorization: Bearer your-secret-api-key-here" http://localhost:8188/prompt
```

**Method 2: X-API-Key Header**
```bash
curl -H "X-API-Key: your-secret-api-key-here" http://localhost:8188/prompt
```

**Method 3: Query Parameter (less secure, for testing only)**
```bash
curl "http://localhost:8188/prompt?api_key=your-secret-api-key-here"
```

#### Exempt Endpoints

The following endpoints do NOT require authentication:
- `/health` - Health check endpoint
- `/` - Root page (frontend)
- `/ws` - WebSocket endpoint

### 2. Health Check Endpoint

A new `/health` endpoint provides server status information.

#### Usage

```bash
curl http://localhost:8188/health
```

#### Response Format

```json
{
  "status": "healthy",
  "version": "0.4.0",
  "timestamp": 1702307890.123,
  "queue": {
    "pending": 0,
    "running": 0
  },
  "device": "cuda:0",
  "vram": {
    "total": 8589934592,
    "free": 6442450944,
    "used": 2147483648
  }
}
```

If the server is unhealthy, it returns a 503 status code:

```json
{
  "status": "unhealthy",
  "error": "error message here",
  "timestamp": 1702307890.123
}
```

## Examples

### Starting ComfyUI with API Key Protection

```bash
# With direct API key
python main.py --api-key "my-super-secret-key-12345"

# With API key from file
python main.py --api-key-file /path/to/api_key.txt

# With API key and custom port
python main.py --api-key "my-key" --port 8080
```

### Making Authenticated Requests

**Python example:**
```python
import requests

API_KEY = "your-api-key-here"
BASE_URL = "http://localhost:8188"

# Using Authorization header
headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# Check health
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Make authenticated request
response = requests.post(
    f"{BASE_URL}/prompt",
    headers=headers,
    json={"prompt": {...}}
)
print(response.json())
```

**JavaScript example:**
```javascript
const API_KEY = "your-api-key-here";
const BASE_URL = "http://localhost:8188";

// Using fetch with Authorization header
async function makeRequest(endpoint, data) {
  const response = await fetch(`${BASE_URL}${endpoint}`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  });
  return response.json();
}

// Check health (no auth required)
fetch(`${BASE_URL}/health`)
  .then(r => r.json())
  .then(data => console.log(data));
```

### Monitoring with Health Check

You can use the health endpoint for monitoring and health checks:

```bash
# Simple health check
curl http://localhost:8188/health

# Use in a monitoring script
#!/bin/bash
response=$(curl -s http://localhost:8188/health)
status=$(echo $response | jq -r '.status')

if [ "$status" == "healthy" ]; then
  echo "✓ ComfyUI is healthy"
  exit 0
else
  echo "✗ ComfyUI is unhealthy"
  exit 1
fi
```

## Security Considerations

1. **Keep your API key secret**: Never commit API keys to version control
2. **Use API key files**: Store API keys in separate files with restricted permissions
3. **Use HTTPS in production**: Combine with `--tls-keyfile` and `--tls-certfile` options
4. **Rotate keys regularly**: Change your API key periodically
5. **Use strong keys**: Generate long, random API keys (e.g., using `openssl rand -hex 32`)

### Generating a Secure API Key

```bash
# Generate a secure random API key
openssl rand -hex 32

# Or using Python
python -c "import secrets; print(secrets.token_hex(32))"
```

## Troubleshooting

### 401 Unauthorized Error

If you receive a 401 error:
- Verify the API key is correct
- Check that you're including the key in the correct header format
- Ensure there are no extra spaces or newlines in the key

### Health Check Returns 503

If the health check returns 503:
- Check the server logs for error details
- Verify ComfyUI started correctly
- Check system resources (memory, disk space)

## Disabling Authentication

To disable API key authentication, simply don't provide the `--api-key` or `--api-key-file` arguments when starting ComfyUI. The server will work exactly as before with no authentication required.
