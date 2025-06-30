"""
Tests for endpoint existence and basic response codes
"""
import pytest
import requests
import logging
import sys
import os
from typing import Dict, Any, List
from urllib.parse import urljoin

# Use a direct import with the full path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Define get_all_endpoints function inline to avoid import issues
def get_all_endpoints(spec):
    """
    Extract all endpoints from an OpenAPI spec

    Args:
        spec: Parsed OpenAPI specification

    Returns:
        List of dicts with path, method, and tags for each endpoint
    """
    endpoints = []

    for path, path_item in spec['paths'].items():
        for method, operation in path_item.items():
            if method.lower() not in ['get', 'post', 'put', 'delete', 'patch']:
                continue

            endpoints.append({
                'path': path,
                'method': method.lower(),
                'tags': operation.get('tags', []),
                'operation_id': operation.get('operationId', ''),
                'summary': operation.get('summary', '')
            })

    return endpoints

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def all_endpoints(api_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all endpoints from the API spec

    Args:
        api_spec: Loaded OpenAPI spec

    Returns:
        List of endpoint information
    """
    return get_all_endpoints(api_spec)


def test_endpoints_exist(all_endpoints: List[Dict[str, Any]]):
    """
    Test that endpoints are defined in the spec

    Args:
        all_endpoints: List of endpoint information
    """
    # Simple check that we have endpoints defined
    assert len(all_endpoints) > 0, "No endpoints defined in the OpenAPI spec"

    # Log the endpoints for informational purposes
    logger.info(f"Found {len(all_endpoints)} endpoints in the OpenAPI spec")
    for endpoint in all_endpoints:
        logger.info(f"{endpoint['method'].upper()} {endpoint['path']} - {endpoint['summary']}")


@pytest.mark.parametrize("endpoint_path", [
    "/",                  # Root path (doesn't have /api prefix)
    "/api/prompt",        # Get prompt info
    "/api/queue",         # Get queue
    "/api/models",        # Get model types
    "/api/object_info",   # Get node info
    "/api/system_stats"   # Get system stats
])
def test_basic_get_endpoints(require_server, api_client, endpoint_path: str):
    """
    Test that basic GET endpoints exist and respond

    Args:
        require_server: Fixture that skips if server is not available
        api_client: API client fixture
        endpoint_path: Path to test
    """
    url = api_client.get_url(endpoint_path)  # type: ignore

    try:
        response = api_client.get(url)

        # We're just checking that the endpoint exists and returns some kind of response
        # Not necessarily a 200 status code
        assert response.status_code not in [404, 405], f"Endpoint {endpoint_path} does not exist"

        logger.info(f"Endpoint {endpoint_path} exists with status code {response.status_code}")

    except requests.RequestException as e:
        pytest.fail(f"Request to {endpoint_path} failed: {str(e)}")


def test_websocket_endpoint_exists(require_server, base_url: str):
    """
    Test that the WebSocket endpoint exists

    Args:
        require_server: Fixture that skips if server is not available
        base_url: Base server URL
    """
    # WebSocket endpoint path from OpenAPI spec
    ws_url = urljoin(base_url, "/api/ws")

    # For WebSocket, we can't use a normal GET request
    # Instead, we make a HEAD request to check if the endpoint exists
    try:
        response = requests.head(ws_url)

        # WebSocket endpoints often return a 400 Bad Request for HEAD requests
        # but a 404 would indicate the endpoint doesn't exist
        assert response.status_code != 404, "WebSocket endpoint /ws does not exist"

        logger.info(f"WebSocket endpoint exists with status code {response.status_code}")

    except requests.RequestException as e:
        pytest.fail(f"Request to WebSocket endpoint failed: {str(e)}")


def test_api_models_folder_endpoint(require_server, api_client):
    """
    Test that the /models/{folder} endpoint exists and responds

    Args:
        require_server: Fixture that skips if server is not available
        api_client: API client fixture
    """
    # First get available model types
    models_url = api_client.get_url("/api/models")  # type: ignore

    try:
        models_response = api_client.get(models_url)
        assert models_response.status_code == 200, "Failed to get model types"

        model_types = models_response.json()

        # Skip if no model types available
        if not model_types:
            pytest.skip("No model types available to test")

        # Test with the first model type
        model_type = model_types[0]
        models_folder_url = api_client.get_url(f"/api/models/{model_type}")  # type: ignore

        folder_response = api_client.get(models_folder_url)

        # We're just checking that the endpoint exists
        assert folder_response.status_code != 404, f"Endpoint /api/models/{model_type} does not exist"

        logger.info(f"Endpoint /api/models/{model_type} exists with status code {folder_response.status_code}")

    except requests.RequestException as e:
        pytest.fail(f"Request failed: {str(e)}")
    except (ValueError, KeyError, IndexError) as e:
        pytest.fail(f"Failed to process response: {str(e)}")


def test_api_object_info_node_endpoint(require_server, api_client):
    """
    Test that the /object_info/{node_class} endpoint exists and responds

    Args:
        require_server: Fixture that skips if server is not available
        api_client: API client fixture
    """
    # First get available node classes
    objects_url = api_client.get_url("/api/object_info")  # type: ignore

    try:
        objects_response = api_client.get(objects_url)
        assert objects_response.status_code == 200, "Failed to get object info"

        node_classes = objects_response.json()

        # Skip if no node classes available
        if not node_classes:
            pytest.skip("No node classes available to test")

        # Test with the first node class
        node_class = next(iter(node_classes.keys()))
        node_url = api_client.get_url(f"/api/object_info/{node_class}")  # type: ignore

        node_response = api_client.get(node_url)

        # We're just checking that the endpoint exists
        assert node_response.status_code != 404, f"Endpoint /api/object_info/{node_class} does not exist"

        logger.info(f"Endpoint /api/object_info/{node_class} exists with status code {node_response.status_code}")

    except requests.RequestException as e:
        pytest.fail(f"Request failed: {str(e)}")
    except (ValueError, KeyError, StopIteration) as e:
        pytest.fail(f"Failed to process response: {str(e)}")


def test_internal_endpoints_exist(require_server, api_client, base_url: str):
    """
    Test that internal endpoints exist

    Args:
        require_server: Fixture that skips if server is not available
        api_client: API client fixture
        base_url: Base server URL
    """
    internal_endpoints = [
        "/internal/logs",
        "/internal/logs/raw",
        "/internal/folder_paths",
        "/internal/files/output"
    ]

    for endpoint in internal_endpoints:
        # Internal endpoints don't use the /api/ prefix
        url = urljoin(base_url, endpoint)

        try:
            response = requests.get(url)

            # We're just checking that the endpoint exists
            assert response.status_code != 404, f"Endpoint {endpoint} does not exist"

            logger.info(f"Endpoint {endpoint} exists with status code {response.status_code}")

        except requests.RequestException as e:
            logger.warning(f"Request to {endpoint} failed: {str(e)}")
            # Don't fail the test as internal endpoints might be restricted
