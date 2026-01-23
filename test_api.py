#!/usr/bin/env python3
"""
Simple API test script for ComfyUI authentication module
"""

import requests
import json

def test_api():
    base_url = "http://localhost:8188"
    
    print("Testing ComfyUI Authentication API...")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    response = requests.get(f"{base_url}/api/auth/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test user registration
    print("\n2. Testing user registration...")
    user_data = {
        "username": "testuser123",
        "email": "test@example.com",
        "password": "Test12345"
    }
    response = requests.post(f"{base_url}/api/auth/register", json=user_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 201:
        print("Registration successful!")
    else:
        print(f"Error: {response.json()}")
    
    # Test duplicate registration
    print("\n3. Testing duplicate registration...")
    response = requests.post(f"{base_url}/api/auth/register", json=user_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test user login
    print("\n4. Testing user login...")
    login_data = {
        "username": "testuser123",
        "password": "Test12345"
    }
    response = requests.post(f"{base_url}/api/auth/login", data=login_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Login successful!")
        tokens = response.json()
        access_token = tokens["access_token"]
        refresh_token = tokens["refresh_token"]
        print(f"Access token: {access_token[:20]}...")
        print(f"Refresh token: {refresh_token[:20]}...")
    else:
        print(f"Error: {response.json()}")
        return
    
    # Test protected route
    print("\n5. Testing protected route...")
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(f"{base_url}/api/auth/protected", headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test get user info
    print("\n6. Testing get user info...")
    response = requests.get(f"{base_url}/api/auth/me", headers=headers)
    print(f"Status: {response.status_code}")
    print(f"User info: {response.json()}")
    
    # Test token refresh
    print("\n7. Testing token refresh...")
    headers = {"Authorization": f"Bearer {refresh_token}"}
    response = requests.post(f"{base_url}/api/auth/refresh", headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        new_tokens = response.json()
        print(f"New access token: {new_tokens['access_token'][:20]}...")
    
    # Test logout
    print("\n8. Testing logout...")
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.post(f"{base_url}/api/auth/logout", headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    print("\n✅ API tests completed!")

if __name__ == "__main__":
    test_api()