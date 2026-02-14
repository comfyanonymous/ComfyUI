"""
Port check utility for ComfyUI
Checks if the server port is available before starting
to avoid port conflicts after custom nodes load
"""

import socket
import sys
import logging


def check_port_available(host, port, timeout=1):
    """
    Check if a port is available for binding.
    
    Args:
        host: Host address to check
        port: Port number to check
        timeout: Connection timeout in seconds
        
    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            if result == 0:
                # Port is in use
                return False
            else:
                # Port is available
                return True
    except socket.error as e:
        logging.error(f"Error checking port {port}: {e}")
        return False


def check_server_port(host, port):
    """
    Check if the server port is available and exit if not.
    This should be called early in the startup process.
    
    Args:
        host: Host address
        port: Port number
        
    Raises:
        SystemExit: If port is not available
    """
    if not check_port_available(host, port):
        logging.error(f"\n{'='*60}")
        logging.error(f"PORT CONFLICT DETECTED")
        logging.error(f"{'='*60}")
        logging.error(f"Port {port} on {host} is already in use.")
        logging.error(f"Please either:")
        logging.error(f"  1. Stop the other process using port {port}")
        logging.error(f"  2. Start ComfyUI on a different port using: --port <PORT>")
        logging.error(f"{'='*60}\n")
        sys.exit(1)
    else:
        logging.info(f"Port {port} is available.")


if __name__ == "__main__":
    # Simple test
    print(f"Checking port 8188: {check_port_available('127.0.0.1', 8188)}")
