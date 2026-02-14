"""
Test for port check functionality in ComfyUI
"""

import socket
import threading
import time
import sys
sys.path.insert(0, '/tmp/comfyui-port-check')

from port_check import check_port_available, check_server_port


def test_port_available():
    """Test that an available port returns True"""
    # Find an available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    
    # Check that it's available
    result = check_port_available('127.0.0.1', port)
    assert result == True, f"Port {port} should be available"
    print(f"✅ Port {port} is available (as expected)")


def test_port_in_use():
    """Test that an in-use port returns False"""
    # Create a socket and bind it
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.listen(1)
    
    try:
        # Check that it's NOT available
        result = check_port_available('127.0.0.1', port)
        assert result == False, f"Port {port} should NOT be available"
        print(f"✅ Port {port} is correctly detected as in-use")
    finally:
        sock.close()


def test_port_check_timeout():
    """Test that port check handles timeout gracefully"""
    # This should not hang
    result = check_port_available('192.0.2.1', 12345, timeout=0.1)
    # Result may vary depending on network, but should not hang
    print(f"✅ Port check completed without hanging")


if __name__ == "__main__":
    print("Testing port check functionality...")
    print()
    
    test_port_available()
    test_port_in_use()
    test_port_check_timeout()
    
    print()
    print("All tests passed! ✅")
