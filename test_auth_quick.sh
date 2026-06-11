#!/bin/bash

# Quick Test Script for ComfyUI API Authentication
# This script tests that authentication is working correctly

set -e

API_KEY="test-key-123"
BASE_URL="http://localhost:8188"

echo "================================================"
echo "ComfyUI API Authentication Test"
echo "================================================"
echo ""
echo "IMPORTANT: Make sure ComfyUI is running with:"
echo "  python main.py --api-key \"$API_KEY\""
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

echo ""
echo "================================================"
echo "Test 1: Health endpoint (should work without auth)"
echo "================================================"
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" "$BASE_URL/health")
status=$(echo "$response" | grep HTTP_STATUS | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

echo "Status: $status"
if [ "$status" = "200" ]; then
    echo "✓ PASS - Health endpoint accessible without auth"
else
    echo "✗ FAIL - Health endpoint should return 200"
fi
echo ""

echo "================================================"
echo "Test 2: Protected endpoint without auth (should fail)"
echo "================================================"
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" "$BASE_URL/object_info")
status=$(echo "$response" | grep HTTP_STATUS | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

echo "Status: $status"
if [ "$status" = "401" ]; then
    echo "✓ PASS - Correctly rejected without auth"
    echo "Response: $body"
else
    echo "✗ FAIL - Should return 401 Unauthorized"
    echo "Response: $body"
fi
echo ""

echo "================================================"
echo "Test 3: Protected endpoint with wrong key (should fail)"
echo "================================================"
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
    -H "Authorization: Bearer wrong-key-456" \
    "$BASE_URL/object_info")
status=$(echo "$response" | grep HTTP_STATUS | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

echo "Status: $status"
if [ "$status" = "401" ]; then
    echo "✓ PASS - Correctly rejected wrong key"
    echo "Response: $body"
else
    echo "✗ FAIL - Should return 401 Unauthorized"
    echo "Response: $body"
fi
echo ""

echo "================================================"
echo "Test 4: Protected endpoint with correct key (should work)"
echo "================================================"
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
    -H "Authorization: Bearer $API_KEY" \
    "$BASE_URL/object_info")
status=$(echo "$response" | grep HTTP_STATUS | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

echo "Status: $status"
if [ "$status" = "200" ]; then
    echo "✓ PASS - Successfully authenticated"
else
    echo "✗ FAIL - Should return 200 OK"
    echo "Response: $body"
fi
echo ""

echo "================================================"
echo "Test 5: X-API-Key header method (should work)"
echo "================================================"
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
    -H "X-API-Key: $API_KEY" \
    "$BASE_URL/embeddings")
status=$(echo "$response" | grep HTTP_STATUS | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

echo "Status: $status"
if [ "$status" = "200" ]; then
    echo "✓ PASS - X-API-Key header works"
else
    echo "✗ FAIL - Should return 200 OK"
    echo "Response: $body"
fi
echo ""

echo "================================================"
echo "Test 6: Query parameter method (should work)"
echo "================================================"
response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
    "$BASE_URL/embeddings?api_key=$API_KEY")
status=$(echo "$response" | grep HTTP_STATUS | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS/d')

echo "Status: $status"
if [ "$status" = "200" ]; then
    echo "✓ PASS - Query parameter works"
else
    echo "✗ FAIL - Should return 200 OK"
    echo "Response: $body"
fi
echo ""

echo "================================================"
echo "All tests completed!"
echo "================================================"
