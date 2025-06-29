"""
Test fixtures for API testing
"""
import os
import pytest
import yaml
import requests
import logging
from typing import Dict, Any, Generator, Optional
from urllib.parse import urljoin

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default server configuration
DEFAULT_SERVER_URL = "http://127.0.0.1:8188"


@pytest.fixture(scope="session")
def api_spec_path() -> str:
    """
    Get the path to the OpenAPI specification file

    Returns:
        Path to the OpenAPI specification file
    """
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "openapi.yaml"
    ))


@pytest.fixture(scope="session")
def api_spec(api_spec_path: str) -> Dict[str, Any]:
    """
    Load the OpenAPI specification

    Args:
        api_spec_path: Path to the spec file

    Returns:
        Parsed OpenAPI specification
    """
    with open(api_spec_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def base_url() -> str:
    """
    Get the base URL for the API server

    Returns:
        Base URL string
    """
    # Allow overriding via environment variable
    return os.environ.get("COMFYUI_SERVER_URL", DEFAULT_SERVER_URL)


@pytest.fixture(scope="session")
def server_available(base_url: str) -> bool:
    """
    Check if the server is available

    Args:
        base_url: Base URL for the API

    Returns:
        True if the server is available, False otherwise
    """
    try:
        response = requests.get(base_url, timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        logger.warning(f"Server at {base_url} is not available")
        return False


@pytest.fixture
def api_client(base_url: str) -> Generator[Optional[requests.Session], None, None]:
    """
    Create a requests session for API testing

    Args:
        base_url: Base URL for the API

    Yields:
        Requests session configured for the API
    """
    session = requests.Session()

    # Helper function to construct URLs
    def get_url(path: str) -> str:
        # All API endpoints use the /api prefix
        return urljoin(base_url, '/api' + path)

    # Add url helper to the session
    session.get_url = get_url  # type: ignore

    yield session

    # Cleanup
    session.close()


@pytest.fixture
def api_get_json(api_client: requests.Session):
    """
    Helper fixture for making GET requests and parsing JSON responses

    Args:
        api_client: API client session

    Returns:
        Function that makes GET requests and returns JSON
    """
    def _get_json(path: str, **kwargs):
        url = api_client.get_url(path)  # type: ignore
        response = api_client.get(url, **kwargs)

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                return None
        return None

    return _get_json


@pytest.fixture
def require_server(server_available):
    """
    Skip tests if server is not available

    Args:
        server_available: Whether the server is available
    """
    if not server_available:
        pytest.skip("Server is not available")
