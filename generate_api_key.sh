#!/bin/bash

# ComfyUI API Key Generator
# This script helps you generate and configure API keys for ComfyUI

set -e

echo "================================================"
echo "ComfyUI API Key Generator"
echo "================================================"
echo ""

# Function to generate a random API key
generate_key() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 32
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "import secrets; print(secrets.token_hex(32))"
    elif command -v python >/dev/null 2>&1; then
        python -c "import secrets; print(secrets.token_hex(32))"
    else
        echo "Error: Neither openssl nor python is available to generate random key"
        exit 1
    fi
}

# Generate the API key
echo "Generating secure API key..."
API_KEY=$(generate_key)
echo ""
echo "Generated API Key:"
echo "================================================"
echo "$API_KEY"
echo "================================================"
echo ""

# Ask user if they want to save to file
read -p "Would you like to save this key to a file? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Get filename
    read -p "Enter filename (default: api_key.txt): " FILENAME
    FILENAME=${FILENAME:-api_key.txt}
    
    # Save the key
    echo "$API_KEY" > "$FILENAME"
    
    # Set restrictive permissions
    chmod 600 "$FILENAME"
    
    echo "✓ API key saved to: $FILENAME"
    echo "✓ File permissions set to 600 (owner read/write only)"
    echo ""
    echo "To start ComfyUI with this API key:"
    echo "  python main.py --api-key-file $FILENAME"
else
    echo ""
    echo "To start ComfyUI with this API key:"
    echo "  python main.py --api-key \"$API_KEY\""
fi

echo ""
echo "================================================"
echo "Important Security Notes:"
echo "================================================"
echo "1. Keep this key secret - don't commit it to git"
echo "2. Use HTTPS in production for encrypted transport"
echo "3. Rotate keys regularly"
echo "4. Add your key file to .gitignore"
echo ""
echo "Example .gitignore entry:"
echo "  api_key.txt"
echo "  *.key"
echo "================================================"
