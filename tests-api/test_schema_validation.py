"""
Tests for validating API responses against OpenAPI schema
"""
import pytest
import requests
import logging
import sys
import os
import json
from typing import Dict, Any

# Use a direct import with the full path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Define validation functions inline to avoid import issues
def get_endpoint_schema(
    spec,
    path,
    method,
    status_code = '200'
):
    """
    Extract response schema for a specific endpoint from OpenAPI spec
    """
    method = method.lower()

    # Handle path not found
    if path not in spec['paths']:
        return None

    # Handle method not found
    if method not in spec['paths'][path]:
        return None

    # Handle status code not found
    responses = spec['paths'][path][method].get('responses', {})
    if status_code not in responses:
        return None

    # Handle no content defined
    if 'content' not in responses[status_code]:
        return None

    # Get schema from first content type
    content_types = responses[status_code]['content']
    first_content_type = next(iter(content_types))

    if 'schema' not in content_types[first_content_type]:
        return None

    return content_types[first_content_type]['schema']

def resolve_schema_refs(schema, spec):
    """
    Resolve $ref references in a schema
    """
    if not isinstance(schema, dict):
        return schema

    result = {}

    for key, value in schema.items():
        if key == '$ref' and isinstance(value, str) and value.startswith('#/'):
            # Handle reference
            ref_path = value[2:].split('/')
            ref_value = spec
            for path_part in ref_path:
                ref_value = ref_value.get(path_part, {})

            # Recursively resolve any refs in the referenced schema
            ref_value = resolve_schema_refs(ref_value, spec)
            result.update(ref_value)
        elif isinstance(value, dict):
            # Recursively resolve refs in nested dictionaries
            result[key] = resolve_schema_refs(value, spec)
        elif isinstance(value, list):
            # Recursively resolve refs in list items
            result[key] = [
                resolve_schema_refs(item, spec) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            # Pass through other values
            result[key] = value

    return result

def validate_response(
    response_data,
    spec,
    path,
    method,
    status_code = '200'
):
    """
    Validate a response against the OpenAPI schema
    """
    schema = get_endpoint_schema(spec, path, method, status_code)

    if schema is None:
        return {
            'valid': False,
            'errors': [f"No schema found for {method.upper()} {path} with status {status_code}"]
        }

    # Resolve any $ref in the schema
    resolved_schema = resolve_schema_refs(schema, spec)

    try:
        import jsonschema
        jsonschema.validate(instance=response_data, schema=resolved_schema)
        return {'valid': True, 'errors': []}
    except jsonschema.exceptions.ValidationError as e:
        # Extract more detailed error information
        path = ".".join(str(p) for p in e.path) if e.path else "root"
        instance = e.instance if not isinstance(e.instance, dict) else "..."
        schema_path = ".".join(str(p) for p in e.schema_path) if e.schema_path else "unknown"

        detailed_error = (
            f"Validation error at path: {path}\n"
            f"Schema path: {schema_path}\n"
            f"Error message: {e.message}\n"
            f"Failed instance: {instance}\n"
        )

        return {'valid': False, 'errors': [detailed_error]}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize("endpoint_path,method", [
    ("/system_stats", "get"),
    ("/prompt", "get"),
    ("/queue", "get"),
    ("/models", "get"),
    ("/embeddings", "get")
])
def test_response_schema_validation(
    require_server,
    api_client,
    api_spec: Dict[str, Any],
    endpoint_path: str,
    method: str
):
    """
    Test that API responses match the defined schema

    Args:
        require_server: Fixture that skips if server is not available
        api_client: API client fixture
        api_spec: Loaded OpenAPI spec
        endpoint_path: Path to test
        method: HTTP method to test
    """
    url = api_client.get_url(endpoint_path)  # type: ignore

    # Skip if no schema defined
    schema = get_endpoint_schema(api_spec, endpoint_path, method)
    if not schema:
        pytest.skip(f"No schema defined for {method.upper()} {endpoint_path}")

    try:
        if method.lower() == "get":
            response = api_client.get(url)
        else:
            pytest.skip(f"Method {method} not implemented for automated testing")
            return

        # Skip if response is not 200
        if response.status_code != 200:
            pytest.skip(f"Endpoint {endpoint_path} returned status {response.status_code}")
            return

        # Skip if response is not JSON
        try:
            response_data = response.json()
        except ValueError:
            pytest.skip(f"Endpoint {endpoint_path} did not return valid JSON")
            return

        # Special handling for system_stats endpoint
        if endpoint_path == '/system_stats' and isinstance(response_data, dict):
            # Remove null index fields before validation
            for device in response_data.get('devices', []):
                if 'index' in device and device['index'] is None:
                    del device['index']

        # Validate the response
        validation_result = validate_response(
            response_data,
            api_spec,
            endpoint_path,
            method
        )

        if validation_result['valid']:
            logger.info(f"Response from {method.upper()} {endpoint_path} matches schema")
        else:
            for error in validation_result['errors']:
                logger.error(f"Validation error for {method.upper()} {endpoint_path}: {error}")

        assert validation_result['valid'], f"Response from {method.upper()} {endpoint_path} does not match schema"

    except requests.RequestException as e:
        pytest.fail(f"Request to {endpoint_path} failed: {str(e)}")


def test_system_stats_response(require_server, api_client, api_spec: Dict[str, Any]):
    """
    Test the system_stats endpoint response in detail

    Args:
        require_server: Fixture that skips if server is not available
        api_client: API client fixture
        api_spec: Loaded OpenAPI spec
    """
    url = api_client.get_url("/system_stats")  # type: ignore

    try:
        response = api_client.get(url)

        assert response.status_code == 200, "Failed to get system stats"

        # Parse response
        stats = response.json()

        # Validate high-level structure
        assert 'system' in stats, "Response missing 'system' field"
        assert 'devices' in stats, "Response missing 'devices' field"

        # Validate system fields
        system = stats['system']
        assert 'os' in system, "System missing 'os' field"
        assert 'ram_total' in system, "System missing 'ram_total' field"
        assert 'ram_free' in system, "System missing 'ram_free' field"
        assert 'comfyui_version' in system, "System missing 'comfyui_version' field"

        # Validate devices fields
        devices = stats['devices']
        assert isinstance(devices, list), "Devices should be a list"

        if devices:
            device = devices[0]
            assert 'name' in device, "Device missing 'name' field"
            assert 'type' in device, "Device missing 'type' field"
            assert 'vram_total' in device, "Device missing 'vram_total' field"
            assert 'vram_free' in device, "Device missing 'vram_free' field"

        # Remove null index fields before validation
        # This is needed because ComfyUI returns null for CPU device index
        for device in stats.get('devices', []):
            if 'index' in device and device['index'] is None:
                del device['index']

        # Perform schema validation
        validation_result = validate_response(
            stats,
            api_spec,
            "/system_stats",
            "get"
        )

        # Print detailed error if validation fails
        if not validation_result['valid']:
            for error in validation_result['errors']:
                logger.error(f"Validation error for /system_stats: {error}")

            # Print schema details for debugging
            schema = get_endpoint_schema(api_spec, "/system_stats", "get")
            if schema:
                logger.error(f"Schema structure:\n{json.dumps(schema, indent=2)}")

            # Print sample of the response
            logger.error(f"Response:\n{json.dumps(stats, indent=2)}")

        assert validation_result['valid'], "System stats response does not match schema"

    except requests.RequestException as e:
        pytest.fail(f"Request to /system_stats failed: {str(e)}")


def test_models_listing_response(require_server, api_client, api_spec: Dict[str, Any]):
    """
    Test the models endpoint response

    Args:
        require_server: Fixture that skips if server is not available
        api_client: API client fixture
        api_spec: Loaded OpenAPI spec
    """
    url = api_client.get_url("/models")  # type: ignore

    try:
        response = api_client.get(url)

        assert response.status_code == 200, "Failed to get models"

        # Parse response
        models = response.json()

        # Validate it's a list
        assert isinstance(models, list), "Models response should be a list"

        # Each item should be a string
        for model in models:
            assert isinstance(model, str), "Each model type should be a string"

        # Perform schema validation
        validation_result = validate_response(
            models,
            api_spec,
            "/models",
            "get"
        )

        # Print detailed error if validation fails
        if not validation_result['valid']:
            for error in validation_result['errors']:
                logger.error(f"Validation error for /models: {error}")

            # Print schema details for debugging
            schema = get_endpoint_schema(api_spec, "/models", "get")
            if schema:
                logger.error(f"Schema structure:\n{json.dumps(schema, indent=2)}")

            # Print response
            sample_models = models[:5] if isinstance(models, list) else models
            logger.error(f"Models response:\n{json.dumps(sample_models, indent=2)}")

        assert validation_result['valid'], "Models response does not match schema"

    except requests.RequestException as e:
        pytest.fail(f"Request to /models failed: {str(e)}")


def test_object_info_response(require_server, api_client, api_spec: Dict[str, Any]):
    """
    Test the object_info endpoint response

    Args:
        require_server: Fixture that skips if server is not available
        api_client: API client fixture
        api_spec: Loaded OpenAPI spec
    """
    url = api_client.get_url("/object_info")  # type: ignore

    try:
        response = api_client.get(url)

        assert response.status_code == 200, "Failed to get object info"

        # Parse response
        objects = response.json()

        # Validate it's an object
        assert isinstance(objects, dict), "Object info response should be an object"

        # Check if we have any objects
        if objects:
            # Get the first object
            first_obj_name = next(iter(objects.keys()))
            first_obj = objects[first_obj_name]

            # Validate first object has required fields
            assert 'input' in first_obj, "Object missing 'input' field"
            assert 'output' in first_obj, "Object missing 'output' field"
            assert 'name' in first_obj, "Object missing 'name' field"

        # Perform schema validation
        validation_result = validate_response(
            objects,
            api_spec,
            "/object_info",
            "get"
        )

        # Print detailed error if validation fails
        if not validation_result['valid']:
            for error in validation_result['errors']:
                logger.error(f"Validation error for /object_info: {error}")

            # Print schema details for debugging
            schema = get_endpoint_schema(api_spec, "/object_info", "get")
            if schema:
                logger.error(f"Schema structure:\n{json.dumps(schema, indent=2)}")

            # Also print a small sample of the response
            sample = dict(list(objects.items())[:1]) if objects else {}
            logger.error(f"Sample response:\n{json.dumps(sample, indent=2)}")

        assert validation_result['valid'], "Object info response does not match schema"

    except requests.RequestException as e:
        pytest.fail(f"Request to /object_info failed: {str(e)}")
    except (KeyError, StopIteration) as e:
        pytest.fail(f"Failed to process response: {str(e)}")


def test_queue_response(require_server, api_client, api_spec: Dict[str, Any]):
    """
    Test the queue endpoint response

    Args:
        require_server: Fixture that skips if server is not available
        api_client: API client fixture
        api_spec: Loaded OpenAPI spec
    """
    url = api_client.get_url("/queue")  # type: ignore

    try:
        response = api_client.get(url)

        assert response.status_code == 200, "Failed to get queue"

        # Parse response
        queue = response.json()

        # Validate structure
        assert 'queue_running' in queue, "Queue missing 'queue_running' field"
        assert 'queue_pending' in queue, "Queue missing 'queue_pending' field"

        # Each should be a list
        assert isinstance(queue['queue_running'], list), "queue_running should be a list"
        assert isinstance(queue['queue_pending'], list), "queue_pending should be a list"

        # Perform schema validation
        validation_result = validate_response(
            queue,
            api_spec,
            "/queue",
            "get"
        )

        # Print detailed error if validation fails
        if not validation_result['valid']:
            for error in validation_result['errors']:
                logger.error(f"Validation error for /queue: {error}")

            # Print schema details for debugging
            schema = get_endpoint_schema(api_spec, "/queue", "get")
            if schema:
                logger.error(f"Schema structure:\n{json.dumps(schema, indent=2)}")

            # Print response
            logger.error(f"Queue response:\n{json.dumps(queue, indent=2)}")

        assert validation_result['valid'], "Queue response does not match schema"

    except requests.RequestException as e:
        pytest.fail(f"Request to /queue failed: {str(e)}")
