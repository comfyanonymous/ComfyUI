from __future__ import annotations

import os
import datetime
import json
import logging
import folder_paths

# Get the logger instance
logger = logging.getLogger(__name__)

def get_log_directory():
    """
    Ensures the API log directory exists within ComfyUI's temp directory
    and returns its path.
    """
    base_temp_dir = folder_paths.get_temp_directory()
    log_dir = os.path.join(base_temp_dir, "api_logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating API log directory {log_dir}: {e}")
        # Fallback to base temp directory if sub-directory creation fails
        return base_temp_dir
    return log_dir

def _format_data_for_logging(data):
    """Helper to format data (dict, str, bytes) for logging."""
    if isinstance(data, bytes):
        try:
            return data.decode('utf-8')  # Try to decode as text
        except UnicodeDecodeError:
            return f"[Binary data of length {len(data)} bytes]"
    elif isinstance(data, (dict, list)):
        try:
            return json.dumps(data, indent=2, ensure_ascii=False)
        except TypeError:
            return str(data) # Fallback for non-serializable objects
    return str(data)

def log_request_response(
    operation_id: str,
    request_method: str,
    request_url: str,
    request_headers: dict | None = None,
    request_params: dict | None = None,
    request_data: any = None,
    response_status_code: int | None = None,
    response_headers: dict | None = None,
    response_content: any = None,
    error_message: str | None = None
):
    """
    Logs API request and response details to a file in the temp/api_logs directory.
    """
    log_dir = get_log_directory()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{operation_id.replace('/', '_').replace(':', '_')}.log"
    filepath = os.path.join(log_dir, filename)

    log_content = []

    log_content.append(f"Timestamp: {datetime.datetime.now().isoformat()}")
    log_content.append(f"Operation ID: {operation_id}")
    log_content.append("-" * 30 + " REQUEST " + "-" * 30)
    log_content.append(f"Method: {request_method}")
    log_content.append(f"URL: {request_url}")
    if request_headers:
        log_content.append(f"Headers:\n{_format_data_for_logging(request_headers)}")
    if request_params:
        log_content.append(f"Params:\n{_format_data_for_logging(request_params)}")
    if request_data:
        log_content.append(f"Data/Body:\n{_format_data_for_logging(request_data)}")

    log_content.append("\n" + "-" * 30 + " RESPONSE " + "-" * 30)
    if response_status_code is not None:
        log_content.append(f"Status Code: {response_status_code}")
    if response_headers:
        log_content.append(f"Headers:\n{_format_data_for_logging(response_headers)}")
    if response_content:
        log_content.append(f"Content:\n{_format_data_for_logging(response_content)}")
    if error_message:
        log_content.append(f"Error:\n{error_message}")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(log_content))
        logger.debug(f"API log saved to: {filepath}")
    except Exception as e:
        logger.error(f"Error writing API log to {filepath}: {e}")

if __name__ == '__main__':
    # Example usage (for testing the logger directly)
    logger.setLevel(logging.DEBUG)
    # Mock folder_paths for direct execution if not running within ComfyUI full context
    if not hasattr(folder_paths, 'get_temp_directory'):
        class MockFolderPaths:
            def get_temp_directory(self):
                # Create a local temp dir for testing if needed
                p = os.path.join(os.path.dirname(__file__), 'temp_test_logs')
                os.makedirs(p, exist_ok=True)
                return p
        folder_paths = MockFolderPaths()

    log_request_response(
        operation_id="test_operation_get",
        request_method="GET",
        request_url="https://api.example.com/test",
        request_headers={"Authorization": "Bearer testtoken"},
        request_params={"param1": "value1"},
        response_status_code=200,
        response_content={"message": "Success!"}
    )
    log_request_response(
        operation_id="test_operation_post_error",
        request_method="POST",
        request_url="https://api.example.com/submit",
        request_data={"key": "value", "nested": {"num": 123}},
        error_message="Connection timed out"
    )
    log_request_response(
        operation_id="test_binary_response",
        request_method="GET",
        request_url="https://api.example.com/image.png",
        response_status_code=200,
        response_content=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...' # Sample binary data
    )
