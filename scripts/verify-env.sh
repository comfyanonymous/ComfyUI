#!/bin/bash
# Verification script to ensure .env file is properly loaded into container
# This script verifies that GOOGLE_API_KEY is accessible in the ComfyUI container

set -e

echo "=== Environment Variable Verification ==="
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found in current directory"
    exit 1
fi
echo "PASS: .env file exists"

# Check if GOOGLE_API_KEY is in .env file
if ! grep -q "^GOOGLE_API_KEY=" .env; then
    echo "ERROR: GOOGLE_API_KEY not found in .env file"
    exit 1
fi
echo "PASS: GOOGLE_API_KEY found in .env file"

# Check docker-compose config
echo ""
echo "=== Docker Compose Configuration ==="
if docker compose config 2>/dev/null | grep -q "GOOGLE_API_KEY"; then
    echo "PASS: GOOGLE_API_KEY found in docker-compose config"
    docker compose config | grep -A 2 "GOOGLE_API_KEY" | head -3
else
    echo "ERROR: GOOGLE_API_KEY not found in docker-compose config"
    exit 1
fi

# Check if container is running
if ! docker compose ps | grep -q "Up"; then
    echo ""
    echo "WARNING: Container is not running. Start it with: docker compose up -d"
    exit 0
fi

echo ""
echo "=== Container Environment Verification ==="

# Check environment variable in container
if docker compose exec -T comfyui env | grep -q "GOOGLE_API_KEY="; then
    echo "PASS: GOOGLE_API_KEY is set in container environment"
    # Show first few characters for verification (don't expose full key)
    API_KEY_PREVIEW=$(docker compose exec -T comfyui env | grep "GOOGLE_API_KEY=" | cut -d'=' -f2 | cut -c1-10)
    echo "   Preview: GOOGLE_API_KEY=${API_KEY_PREVIEW}..."
else
    echo "ERROR: GOOGLE_API_KEY not found in container environment"
    exit 1
fi

# Verify Python can access it
echo ""
echo "=== Python Environment Access ==="
if docker compose exec -T comfyui python -c "import os; key = os.environ.get('GOOGLE_API_KEY'); print('PASS: Python can access GOOGLE_API_KEY:', 'YES' if key else 'NO')" 2>/dev/null | grep -q "YES"; then
    echo "PASS: Python can access GOOGLE_API_KEY via os.environ"
else
    echo "ERROR: Python cannot access GOOGLE_API_KEY"
    exit 1
fi

echo ""
echo "=== Verification Complete ==="
echo "PASS: All checks passed! GOOGLE_API_KEY is properly configured."
echo ""
echo "To test Nano Banana authentication:"
echo "  1. Ensure Nano Banana is installed in custom_nodes/"
echo "  2. Access ComfyUI at http://localhost:8189"
echo "  3. Check logs: docker compose logs comfyui | grep -i 'nano\|banana\|credentials'"

