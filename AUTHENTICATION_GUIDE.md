# ComfyUI API Authentication Guide

## Overview

ComfyUI now supports API key authentication to protect your REST API endpoints. This guide covers setup, usage, and logout procedures.

## Features

- **API Key Authentication**: Protect all API endpoints with a simple API key
- **Multiple Auth Methods**: Bearer token, X-API-Key header, or query parameter
- **Health Endpoint**: Monitor server status without authentication
- **Frontend Login**: Built-in login page for browser access
- **Session Management**: Remember API key across sessions or per-session only

## Quick Start

### 1. Start Server with API Key

```bash
# Using command line argument
python3 main.py --api-key "your-secure-api-key-here"

# Or using a file (recommended for production)
echo "your-secure-api-key-here" > api_key.txt
python3 main.py --api-key-file api_key.txt
```

### 2. Access ComfyUI

When you navigate to `http://127.0.0.1:8188`, you'll be redirected to a login page.

### 3. Login

1. Enter your API key in the password field
2. Optionally check "Remember me" to persist the key across browser sessions
3. Click "Login"

The system validates your key against the `/health` endpoint before storing it.

### 4. Logout

To logout, visit:
```
http://127.0.0.1:8188/auth_login.html?logout=true
```

This will clear your stored API key and redirect you to the login page.

## API Usage

### Authentication Methods

You can authenticate API requests in three ways:

#### 1. Bearer Token (Recommended)
```bash
curl -H "Authorization: Bearer your-api-key" http://127.0.0.1:8188/prompt
```

#### 2. X-API-Key Header
```bash
curl -H "X-API-Key: your-api-key" http://127.0.0.1:8188/prompt
```

#### 3. Query Parameter
```bash
curl "http://127.0.0.1:8188/prompt?api_key=your-api-key"
```

### Python Example

```python
import requests

API_KEY = "your-api-key"
BASE_URL = "http://127.0.0.1:8188"

# Using Bearer token
headers = {"Authorization": f"Bearer {API_KEY}"}
response = requests.get(f"{BASE_URL}/queue", headers=headers)

# Using X-API-Key header
headers = {"X-API-Key": API_KEY}
response = requests.get(f"{BASE_URL}/queue", headers=headers)

# Using query parameter
response = requests.get(f"{BASE_URL}/queue?api_key={API_KEY}")
```

### JavaScript Example

```javascript
const API_KEY = "your-api-key";
const BASE_URL = "http://127.0.0.1:8188";

// Using Bearer token
fetch(`${BASE_URL}/queue`, {
    headers: {
        "Authorization": `Bearer ${API_KEY}`
    }
})
.then(response => response.json())
.then(data => console.log(data));
```

## Health Endpoint

The `/health` endpoint is always accessible without authentication:

```bash
curl http://127.0.0.1:8188/health
```

Response includes:
- Server status
- Queue information (pending/running tasks)
- Device information (GPU/CPU)
- VRAM statistics

Example response:
```json
{
  "status": "ok",
  "queue": {
    "pending": 0,
    "running": 0
  },
  "device": {
    "name": "mps",
    "type": "mps"
  },
  "vram": {
    "total": 68719476736,
    "free": 68719476736
  }
}
```

## Exempt Endpoints

The following paths are exempt from authentication:

### Static Files
- `.html`, `.js`, `.css`, `.json` files
- `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` images
- `.svg`, `.ico` icons
- `.woff`, `.woff2`, `.ttf` fonts
- `.mp3`, `.wav` audio files
- `.mp4`, `.webm` video files

### API Paths
- `/` - Root path (serves login if not authenticated)
- `/health` - Health check endpoint
- `/ws` - WebSocket endpoint
- `/auth_login.html` - Login page
- `/auth_inject.js` - Auth injection script
- `/extensions/*` - Extension files
- `/templates/*` - Template files
- `/docs/*` - Documentation

## Security Best Practices

1. **Use Strong API Keys**: Generate random, long API keys (32+ characters)
   ```bash
   # Generate secure API key on macOS/Linux
   openssl rand -base64 32
   ```

2. **Use API Key Files**: Store keys in files rather than command line
   ```bash
   python3 main.py --api-key-file /secure/path/api_key.txt
   ```

3. **File Permissions**: Restrict key file access
   ```bash
   chmod 600 api_key.txt
   ```

4. **HTTPS**: Use reverse proxy with SSL in production
   ```nginx
   server {
       listen 443 ssl;
       server_name your-domain.com;
       
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
       
       location / {
           proxy_pass http://127.0.0.1:8188;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

5. **Environment Variables**: Store keys in environment
   ```bash
   export COMFYUI_API_KEY=$(cat api_key.txt)
   python3 main.py --api-key "$COMFYUI_API_KEY"
   ```

## Frontend Integration

The authentication system automatically handles frontend requests:

1. When authentication is enabled, `auth_inject.js` is injected into `index.html`
2. This script intercepts all `fetch()` and `XMLHttpRequest` calls
3. Authorization headers are added automatically to all requests
4. On 401 responses, the user is redirected to the login page

### Session Storage

- **localStorage**: API key persists across browser sessions (when "Remember me" is checked)
- **sessionStorage**: API key cleared when browser/tab closes (when "Remember me" is not checked)

## Troubleshooting

### 401 Unauthorized Errors

1. Verify API key matches server configuration
2. Check authentication header format
3. Ensure endpoint isn't expecting different auth method

### Login Page Not Appearing

1. Clear browser cache
2. Verify `auth_login.html` and `auth_inject.js` exist in ComfyUI root
3. Check server logs for errors

### WebSocket Connection Issues

WebSocket connections (`/ws`) are exempt from authentication, but may require authentication for initial HTTP upgrade depending on your setup.

### Logout Not Working

Visit the logout URL directly:
```
http://127.0.0.1:8188/auth_login.html?logout=true
```

Or clear storage manually in browser DevTools:
```javascript
localStorage.removeItem('comfyui_api_key');
sessionStorage.removeItem('comfyui_api_key');
```

## Disabling Authentication

To disable authentication, simply start the server without the `--api-key` or `--api-key-file` arguments:

```bash
python3 main.py
```

## Migration from Unauthenticated Setup

If you're adding authentication to an existing ComfyUI installation:

1. Ensure `auth_login.html` and `auth_inject.js` exist in root directory
2. Update `server.py` with authentication routes
3. Add `middleware/auth_middleware.py`
4. Update `comfy/cli_args.py` with API key arguments
5. Restart server with `--api-key` argument
6. Update API clients to include authentication

## Support

For issues or questions:
- Check server logs for authentication errors
- Verify middleware is properly configured
- Test with `/health` endpoint first (no auth required)
- Review `middleware/auth_middleware.py` for exempt paths
