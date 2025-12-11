#!/usr/bin/env python3
"""
Test script for ComfyUI API Key Authentication and Health Check

This script demonstrates how to:
1. Check the health endpoint (no auth required)
2. Make authenticated requests to the API
"""

import requests
import json
import sys

# Configuration
BASE_URL = "http://localhost:8188"
API_KEY = "your-api-key-here"  # Replace with your actual API key


def test_health_check():
    """Test the health check endpoint (no authentication required)"""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_without_auth():
    """Test accessing protected endpoint without authentication"""
    print("\nTesting access without authentication...")
    try:
        response = requests.get(f"{BASE_URL}/object_info")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 401:
            print("✓ Correctly rejected (401 Unauthorized)")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        elif response.status_code == 200:
            print("✓ No authentication required (API key not enabled)")
            return True
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_with_bearer_token():
    """Test accessing protected endpoint with Bearer token"""
    print("\nTesting with Bearer token authentication...")
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}"
        }
        response = requests.get(f"{BASE_URL}/object_info", headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("✓ Successfully authenticated with Bearer token")
            return True
        elif response.status_code == 401:
            print("✗ Authentication failed (check your API key)")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return False
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_with_api_key_header():
    """Test accessing protected endpoint with X-API-Key header"""
    print("\nTesting with X-API-Key header authentication...")
    try:
        headers = {
            "X-API-Key": API_KEY
        }
        response = requests.get(f"{BASE_URL}/object_info", headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("✓ Successfully authenticated with X-API-Key header")
            return True
        elif response.status_code == 401:
            print("✗ Authentication failed (check your API key)")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return False
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_with_query_parameter():
    """Test accessing protected endpoint with query parameter"""
    print("\nTesting with query parameter authentication...")
    try:
        response = requests.get(f"{BASE_URL}/object_info?api_key={API_KEY}")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("✓ Successfully authenticated with query parameter")
            return True
        elif response.status_code == 401:
            print("✗ Authentication failed (check your API key)")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return False
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("ComfyUI API Authentication Test Suite")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"API Key: {'*' * (len(API_KEY) - 4) + API_KEY[-4:] if len(API_KEY) > 4 else '***'}")
    print("=" * 60)
    
    results = []
    
    # Test 1: Health check (always works)
    results.append(("Health Check", test_health_check()))
    
    # Test 2: Without authentication (should fail if auth is enabled)
    results.append(("No Auth", test_without_auth()))
    
    # Test 3: Bearer token authentication
    results.append(("Bearer Token", test_with_bearer_token()))
    
    # Test 4: X-API-Key header authentication
    results.append(("X-API-Key Header", test_with_api_key_header()))
    
    # Test 5: Query parameter authentication
    results.append(("Query Parameter", test_with_query_parameter()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")
    
    total = len(results)
    passed = sum(1 for _, result in results if result)
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    # Check if user wants to override the API key
    if len(sys.argv) > 1:
        API_KEY = sys.argv[1]
    
    if API_KEY == "your-api-key-here":
        print("WARNING: Using default API key. Set your API key as the first argument:")
        print(f"  python {sys.argv[0]} YOUR_API_KEY")
        print("")
    
    main()
