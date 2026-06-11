# ComfyUI Frontend Authentication Guide

## Overview

When API key authentication is enabled, the ComfyUI frontend will automatically require users to log in with the API key before accessing the interface.

## How It Works

1. **Automatic Redirection**: When you access ComfyUI with authentication enabled, you'll be automatically redirected to a login page if no valid API key is stored.

2. **API Key Storage**: After successful login, your API key is stored in your browser (localStorage or sessionStorage) depending on your "Remember me" choice.

3. **Automatic Injection**: The API key is automatically added to all API requests via the `Authorization: Bearer <key>` header.

4. **Session Management**: 
   - **Remember me (checked)**: API key stored in localStorage (persists across browser sessions)
   - **Remember me (unchecked)**: API key stored in sessionStorage (cleared when browser closes)

## Using the Frontend with Authentication

### First Time Access

1. Start ComfyUI with an API key:
   ```bash
   python main.py --api-key "your-secret-key-123"
   ```

2. Open your browser and navigate to `http://localhost:8188`

3. You'll be presented with a login page

4. Enter your API key and click "Login"

5. Choose whether to remember your login (recommended for personal devices only)

### Login Page Features

- **Show API Key**: Toggle to view the key you're entering
- **Remember Me**: Keep your session active across browser restarts
- **Auto-validation**: The system validates your key before storing it

### Logging Out

To log out and clear your stored API key:

**Option 1: JavaScript Console**
```javascript
comfyuiLogout()
```

**Option 2: Browser DevTools**
1. Open DevTools (F12)
2. Go to Application > Storage
3. Clear localStorage and sessionStorage
4. Refresh the page

**Option 3: Manual URL**
Navigate to: `http://localhost:8188/auth_login.html`

### Security Considerations

#### For Personal Use
- ✅ Enable "Remember me" for convenience
- ✅ Use strong, unique API keys
- ✅ Keep your browser updated

#### For Shared/Public Computers
- ❌ **DO NOT** enable "Remember me"
- ✅ Always log out when finished
- ✅ Clear browser data after use
- ✅ Consider using a private/incognito window

#### For Production Deployments
- ✅ Always use HTTPS (combine with `--tls-keyfile` and `--tls-certfile`)
- ✅ Use strong, randomly generated API keys
- ✅ Rotate keys regularly
- ✅ Monitor access logs for unauthorized attempts
- ✅ Consider using additional security layers (VPN, firewall, etc.)

## Troubleshooting

### "Invalid API Key" Error

**Problem**: Login fails with invalid API key message

**Solutions**:
1. Verify you're using the correct API key
2. Check for extra spaces or newlines
3. Ensure the server was started with the same API key
4. Check server logs for authentication attempts

### Automatic Logout

**Problem**: Frequently logged out automatically

**Possible Causes**:
1. API key changed on server (restart required)
2. Browser cleared storage automatically
3. "Remember me" was not checked
4. Session expired (if using sessionStorage)

**Solution**: Check "Remember me" when logging in

### Login Page Not Showing

**Problem**: Can't access login page

**Possible Causes**:
1. Authentication not enabled (no `--api-key` argument)
2. Browser cached old version

**Solution**:
1. Hard refresh the page (Ctrl+Shift+R or Cmd+Shift+R)
2. Clear browser cache
3. Try accessing directly: `http://localhost:8188/auth_login.html`

### API Requests Still Failing

**Problem**: Some API requests return 401 after login

**Solutions**:
1. Check browser console for errors
2. Verify API key is stored: Open DevTools > Application > Storage
3. Try logging out and back in
4. Clear all browser data and try again

## Advanced Usage

### Programmatic Access with Frontend

If you need to access the API programmatically while using the frontend:

```javascript
// Get the stored API key
const apiKey = localStorage.getItem('comfyui_api_key') || 
               sessionStorage.getItem('comfyui_api_key');

// Make authenticated requests
fetch('/api/system_stats', {
    headers: {
        'Authorization': `Bearer ${apiKey}`
    }
}).then(r => r.json()).then(console.log);
```

### Custom Frontend Integration

If you're building a custom frontend, you can use the same mechanism:

1. **Login Flow**:
   ```javascript
   // Validate API key
   const response = await fetch('/health', {
       headers: {
           'Authorization': `Bearer ${apiKey}`
       }
   });
   
   if (response.ok) {
       // Store the key
       localStorage.setItem('comfyui_api_key', apiKey);
   }
   ```

2. **Request Interceptor**:
   ```javascript
   // Add to all requests
   const apiKey = localStorage.getItem('comfyui_api_key');
   
   fetch(url, {
       headers: {
           'Authorization': `Bearer ${apiKey}`,
           ...otherHeaders
       }
   });
   ```

3. **Handle 401 Responses**:
   ```javascript
   if (response.status === 401) {
       // Clear stored key and redirect to login
       localStorage.removeItem('comfyui_api_key');
       window.location.href = '/auth_login.html';
   }
   ```

## API Endpoints Reference

### Public Endpoints (No Authentication Required)
- `GET /` - Main page (with auth script injected)
- `GET /health` - Health check
- `GET /auth_login.html` - Login page
- `GET /auth_inject.js` - Auth injection script
- `GET /ws` - WebSocket connection
- Static files (`.js`, `.css`, `.html`, etc.)

### Protected Endpoints (Authentication Required)
- All `/api/*` endpoints
- All `/internal/*` endpoints
- Most other API endpoints

## Browser Compatibility

The authentication system works with all modern browsers:
- ✅ Chrome/Edge 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Opera 76+

## FAQs

**Q: Can I use ComfyUI without authentication?**  
A: Yes! Simply start ComfyUI without the `--api-key` argument.

**Q: Can I change the API key without losing my workflows?**  
A: Yes, workflows are stored separately. Just update the key on the server and re-login.

**Q: Is my API key secure?**  
A: The key is stored in browser storage and sent over HTTPS (if configured). For maximum security, use HTTPS and strong keys.

**Q: Can multiple users use different API keys?**  
A: Currently, the system supports a single API key. For multi-user scenarios, each user must use the same key.

**Q: What happens if I forget my API key?**  
A: Check your server startup command or the file specified in `--api-key-file`.

## Support

For issues or questions:
1. Check the server logs
2. Review browser console for errors
3. Refer to the main authentication documentation: `API_AUTHENTICATION.md`
4. Check the quick start guide: `QUICK_START_AUTH.md`
