# Quick Start Guide - API Authentication

## Step-by-Step Instructions

### 1. Start ComfyUI with API Key

```bash
# Stop any running ComfyUI instance first
# Then start with an API key:

python main.py --api-key "my-secret-key-123"
```

**You should see in the logs:**
```
[Auth] API Key authentication enabled
```

### 2. Test the Authentication

**Health check (works without auth):**
```bash
curl http://localhost:8188/health
```

**Protected endpoint without auth (should fail):**
```bash
curl http://localhost:8188/object_info
# Should return: {"error": "Unauthorized", "message": "..."}
```

**Protected endpoint with auth (should work):**
```bash
curl -H "Authorization: Bearer my-secret-key-123" http://localhost:8188/object_info
# Should return: {...node definitions...}
```

### 3. Run the Test Script

```bash
chmod +x test_auth_quick.sh
./test_auth_quick.sh
```

## Common Issues

### Issue: All requests work without authentication

**Problem:** You didn't start the server with `--api-key`

**Solution:**
```bash
# Stop the server (Ctrl+C)
# Restart with API key:
python main.py --api-key "your-key-here"
```

**Verify it's enabled:**
```bash
# In another terminal, check if auth is working:
curl http://localhost:8188/object_info
# Should return 401 Unauthorized
```

### Issue: Authentication is enabled but I get 401 even with correct key

**Problem:** Key format or typo

**Solution:**
- Ensure no extra spaces in the key
- Check the Authorization header format: `Authorization: Bearer YOUR_KEY`
- Try X-API-Key header: `X-API-Key: YOUR_KEY`

## Example: Full Workflow

```bash
# 1. Generate a secure key
python -c "import secrets; print(secrets.token_hex(32))"
# Output: a1b2c3d4e5f6...

# 2. Save to file
echo "a1b2c3d4e5f6..." > api_key.txt

# 3. Start server with key file
python main.py --api-key-file api_key.txt

# 4. Use the API
API_KEY=$(cat api_key.txt)
curl -H "Authorization: Bearer $API_KEY" http://localhost:8188/object_info
```

## Test with Python

```python
import requests

API_KEY = "my-secret-key-123"
BASE_URL = "http://localhost:8188"

# This should fail (no auth)
response = requests.get(f"{BASE_URL}/object_info")
print(f"No auth: {response.status_code}")  # Should be 401

# This should work (with auth)
headers = {"Authorization": f"Bearer {API_KEY}"}
response = requests.get(f"{BASE_URL}/object_info", headers=headers)
print(f"With auth: {response.status_code}")  # Should be 200
```

## Disable Authentication

Simply start ComfyUI without the `--api-key` argument:

```bash
python main.py
```

The server will work exactly as before with no authentication required.
