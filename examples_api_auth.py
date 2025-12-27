"""
Example: Using ComfyUI API with Authentication
"""

import requests
import json

# Your API configuration
API_KEY = "your-api-key-here"
BASE_URL = "http://localhost:8188"


def example_health_check():
    """Example: Check server health (no authentication required)"""
    print("=== Health Check Example ===")
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        health = response.json()
        print(f"Status: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Queue - Pending: {health['queue']['pending']}, Running: {health['queue']['running']}")
        if 'device' in health:
            print(f"Device: {health['device']}")
        if 'vram' in health:
            vram = health['vram']
            vram_used_gb = vram['used'] / (1024**3)
            vram_total_gb = vram['total'] / (1024**3)
            print(f"VRAM: {vram_used_gb:.2f} GB / {vram_total_gb:.2f} GB")
    else:
        print(f"Health check failed with status {response.status_code}")
    
    print()


def example_get_object_info():
    """Example: Get object info with authentication"""
    print("=== Get Object Info Example ===")
    
    # Method 1: Using Authorization Bearer header
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    response = requests.get(f"{BASE_URL}/object_info", headers=headers)
    
    if response.status_code == 200:
        print("✓ Successfully retrieved object info")
        object_info = response.json()
        print(f"Number of node types: {len(object_info)}")
    elif response.status_code == 401:
        print("✗ Authentication failed - check your API key")
        print(response.json())
    else:
        print(f"✗ Request failed with status {response.status_code}")
    
    print()


def example_queue_prompt():
    """Example: Queue a prompt with authentication"""
    print("=== Queue Prompt Example ===")
    
    # Simple workflow example
    workflow = {
        "prompt": {
            "1": {
                "inputs": {
                    "text": "a beautiful landscape"
                },
                "class_type": "CLIPTextEncode"
            }
        },
        "client_id": "example_client"
    }
    
    # Using Authorization Bearer header
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        f"{BASE_URL}/prompt",
        headers=headers,
        json=workflow
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Prompt queued successfully")
        print(f"Prompt ID: {result.get('prompt_id', 'N/A')}")
    elif response.status_code == 401:
        print("✗ Authentication failed - check your API key")
        print(response.json())
    else:
        print(f"✗ Request failed with status {response.status_code}")
        print(response.text)
    
    print()


def example_using_session():
    """Example: Using requests.Session for multiple requests"""
    print("=== Session Example (Multiple Requests) ===")
    
    # Create a session with authentication header
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {API_KEY}"
    })
    
    # Now all requests will automatically include the auth header
    
    # Request 1: Get embeddings
    response = session.get(f"{BASE_URL}/embeddings")
    if response.status_code == 200:
        print(f"✓ Got embeddings list")
    
    # Request 2: Get queue
    response = session.get(f"{BASE_URL}/queue")
    if response.status_code == 200:
        queue = response.json()
        print(f"✓ Got queue info - Pending: {len(queue.get('queue_pending', []))}")
    
    # Request 3: Get system stats
    response = session.get(f"{BASE_URL}/system_stats")
    if response.status_code == 200:
        print(f"✓ Got system stats")
    
    print()


def example_error_handling():
    """Example: Proper error handling"""
    print("=== Error Handling Example ===")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        response = requests.get(f"{BASE_URL}/queue", headers=headers, timeout=5)
        response.raise_for_status()  # Raises exception for 4xx/5xx status codes
        
        data = response.json()
        print("✓ Request successful")
        print(f"Queue pending: {len(data.get('queue_pending', []))}")
        print(f"Queue running: {len(data.get('queue_running', []))}")
        
    except requests.exceptions.Timeout:
        print("✗ Request timed out")
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to server")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("✗ Authentication failed - invalid API key")
        elif e.response.status_code == 403:
            print("✗ Access forbidden")
        else:
            print(f"✗ HTTP error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    print()


# Alternative authentication methods
def example_alternative_auth_methods():
    """Example: Different ways to provide API key"""
    print("=== Alternative Authentication Methods ===")
    
    # Method 1: Authorization Bearer token (recommended)
    headers1 = {"Authorization": f"Bearer {API_KEY}"}
    response1 = requests.get(f"{BASE_URL}/embeddings", headers=headers1)
    print(f"Method 1 (Bearer): Status {response1.status_code}")
    
    # Method 2: X-API-Key header
    headers2 = {"X-API-Key": API_KEY}
    response2 = requests.get(f"{BASE_URL}/embeddings", headers=headers2)
    print(f"Method 2 (X-API-Key): Status {response2.status_code}")
    
    # Method 3: Query parameter (less secure, not recommended for production)
    response3 = requests.get(f"{BASE_URL}/embeddings?api_key={API_KEY}")
    print(f"Method 3 (Query param): Status {response3.status_code}")
    
    print()


if __name__ == "__main__":
    print("ComfyUI API Authentication Examples")
    print("=" * 60)
    print()
    
    # Run examples
    example_health_check()
    example_get_object_info()
    example_using_session()
    example_error_handling()
    example_alternative_auth_methods()
    
    print("=" * 60)
    print("All examples completed!")
