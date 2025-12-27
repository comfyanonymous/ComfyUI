# ComfyUI API Security Enhancement

## Summary

This implementation adds API key authentication and a health check endpoint to ComfyUI.

## Files Modified

1. **middleware/auth_middleware.py** (NEW)
   - API key authentication middleware
   - Supports multiple authentication methods (Bearer token, X-API-Key header, query parameter)
   - Configurable exempt paths

2. **comfy/cli_args.py** (MODIFIED)
   - Added `--api-key` argument for inline API key
   - Added `--api-key-file` argument for API key from file
   - Added logic to load API key from file

3. **server.py** (MODIFIED)
   - Imported auth middleware
   - Integrated middleware into application
   - Added `/health` endpoint with system information
   - Configured exempt paths (/, /health, /ws)

## New Files

1. **API_AUTHENTICATION.md** - Complete documentation
2. **test_api_auth.py** - Test suite for authentication
3. **examples_api_auth.py** - Python usage examples

## Quick Start

### 1. Start ComfyUI with API Key Protection

```bash
# Generate a secure API key
python -c "import secrets; print(secrets.token_hex(32))"

# Start with API key
python main.py --api-key "your-generated-key-here"

# Or use a file
echo "your-generated-key-here" > api_key.txt
python main.py --api-key-file api_key.txt
```

### 2. Test the Health Endpoint

```bash
curl http://localhost:8188/health
```

### 3. Make Authenticated Requests

```bash
# Using Bearer token
curl -H "Authorization: Bearer your-api-key" http://localhost:8188/prompt

# Using X-API-Key header
curl -H "X-API-Key: your-api-key" http://localhost:8188/prompt
```

### 4. Run Tests

```bash
# Install requests if needed
pip install requests

# Run test suite
python test_api_auth.py your-api-key

# Run examples
python examples_api_auth.py
```

## Features

### API Key Authentication
- ✅ Multiple authentication methods (Bearer, X-API-Key, query param)
- ✅ Configurable via command line
- ✅ Secure file-based configuration
- ✅ Exempt paths for health checks and WebSocket
- ✅ Detailed logging of authentication attempts

### Health Check Endpoint
- ✅ Returns server status
- ✅ Queue information (pending/running)
- ✅ Device information
- ✅ VRAM usage (if GPU available)
- ✅ Version information
- ✅ Timestamp for monitoring

## Security Best Practices

1. **Generate Strong Keys**: Use `openssl rand -hex 32` or similar
2. **Use File-Based Config**: Keep keys out of command history
3. **Enable HTTPS**: Use with `--tls-keyfile` and `--tls-certfile`
4. **Restrict File Permissions**: `chmod 600 api_key.txt`
5. **Rotate Keys Regularly**: Change API keys periodically
6. **Monitor Access**: Check logs for unauthorized attempts

## Backward Compatibility

- ✅ Fully backward compatible
- ✅ No authentication required by default
- ✅ Existing functionality unchanged
- ✅ WebSocket connections work normally

## Testing

The implementation has been tested for:
- ✅ Syntax errors (none found)
- ✅ Import compatibility
- ✅ Middleware integration
- ✅ Route configuration
- ✅ Health endpoint functionality

To fully test in your environment:
```bash
# 1. Start server without auth (test backward compatibility)
python main.py

# 2. Start server with auth
python main.py --api-key "test-key-123"

# 3. Run test suite
python test_api_auth.py test-key-123

# 4. Check health endpoint
curl http://localhost:8188/health
```

## Support

For detailed documentation, see:
- **API_AUTHENTICATION.md** - Complete usage guide
- **examples_api_auth.py** - Code examples
- **test_api_auth.py** - Test suite

## License

Same as ComfyUI main project.
