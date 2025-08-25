#!/usr/bin/env python3
import socket
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing port conflict detection...")

sock_test = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock_test.bind(('', 8188))
    print("✓ Port 8188 is available")
    sock_test.close()
except OSError:
    print("✗ Port 8188 is already in use")

print("\nSimulating standalone build behavior...")
port = 8188
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(('', port))
    sock.close()
    print(f"✓ Would use port {port}")
except OSError:
    print(f"✗ Port {port} is in use, switching to random port (0)")
    port = 0
    
print(f"\nFinal port selection: {port}")

if port == 0:
    test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    test_sock.bind(('', 0))
    actual_port = test_sock.getsockname()[1]
    print(f"Random port assigned: {actual_port}")
    test_sock.close()