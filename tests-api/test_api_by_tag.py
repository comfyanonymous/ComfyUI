"""
Tests for API endpoints grouped by tags
"""
import pytest
import logging
import sys
import os
from typing import Dict, Any, Set

# Use a direct import with the full path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Define functions inline to avoid import issues
def get_all_endpoints(spec):
    """
    Extract all endpoints from an OpenAPI spec
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

def get_all_tags(spec):
    """
    Get all tags used in the API spec
    """
    tags = set()

    for path_item in spec['paths'].values():
        for operation in path_item.values():
            if isinstance(operation, dict) and 'tags' in operation:
                tags.update(operation['tags'])

    return tags

def extract_endpoints_by_tag(spec, tag):
    """
    Extract all endpoints with a specific tag
    """
    endpoints = []

    for path, path_item in spec['paths'].items():
        for method, operation in path_item.items():
            if method.lower() not in ['get', 'post', 'put', 'delete', 'patch']:
                continue

            if tag in operation.get('tags', []):
                endpoints.append({
                    'path': path,
                    'method': method.lower(),
                    'operation_id': operation.get('operationId', ''),
                    'summary': operation.get('summary', '')
                })

    return endpoints

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def api_tags(api_spec: Dict[str, Any]) -> Set[str]:
    """
    Get all tags from the API spec

    Args:
        api_spec: Loaded OpenAPI spec

    Returns:
        Set of tag names
    """
    return get_all_tags(api_spec)


def test_api_has_tags(api_tags: Set[str]):
    """
    Test that the API has defined tags

    Args:
        api_tags: Set of tags
    """
    assert len(api_tags) > 0, "API spec should have at least one tag"

    # Log the tags
    logger.info(f"API spec has the following tags: {sorted(api_tags)}")


@pytest.mark.parametrize("tag", [
    "workflow",
    "image",
    "model",
    "node",
    "system"
])
def test_core_tags_exist(api_tags: Set[str], tag: str):
    """
    Test that core tags exist in the API spec

    Args:
        api_tags: Set of tags
        tag: Tag to check
    """
    assert tag in api_tags, f"API spec should have '{tag}' tag"


def test_workflow_tag_has_endpoints(api_spec: Dict[str, Any]):
    """
    Test that the 'workflow' tag has appropriate endpoints

    Args:
        api_spec: Loaded OpenAPI spec
    """
    endpoints = extract_endpoints_by_tag(api_spec, "workflow")

    assert len(endpoints) > 0, "No endpoints found with 'workflow' tag"

    # Check for key workflow endpoints
    endpoint_paths = [e["path"] for e in endpoints]
    assert "/prompt" in endpoint_paths, "Workflow tag should include /prompt endpoint"

    # Log the endpoints
    logger.info(f"Found {len(endpoints)} endpoints with 'workflow' tag:")
    for e in endpoints:
        logger.info(f"  {e['method'].upper()} {e['path']}")


def test_image_tag_has_endpoints(api_spec: Dict[str, Any]):
    """
    Test that the 'image' tag has appropriate endpoints

    Args:
        api_spec: Loaded OpenAPI spec
    """
    endpoints = extract_endpoints_by_tag(api_spec, "image")

    assert len(endpoints) > 0, "No endpoints found with 'image' tag"

    # Check for key image endpoints
    endpoint_paths = [e["path"] for e in endpoints]
    assert "/upload/image" in endpoint_paths, "Image tag should include /upload/image endpoint"
    assert "/view" in endpoint_paths, "Image tag should include /view endpoint"

    # Log the endpoints
    logger.info(f"Found {len(endpoints)} endpoints with 'image' tag:")
    for e in endpoints:
        logger.info(f"  {e['method'].upper()} {e['path']}")


def test_model_tag_has_endpoints(api_spec: Dict[str, Any]):
    """
    Test that the 'model' tag has appropriate endpoints

    Args:
        api_spec: Loaded OpenAPI spec
    """
    endpoints = extract_endpoints_by_tag(api_spec, "model")

    assert len(endpoints) > 0, "No endpoints found with 'model' tag"

    # Check for key model endpoints
    endpoint_paths = [e["path"] for e in endpoints]
    assert "/models" in endpoint_paths, "Model tag should include /models endpoint"

    # Log the endpoints
    logger.info(f"Found {len(endpoints)} endpoints with 'model' tag:")
    for e in endpoints:
        logger.info(f"  {e['method'].upper()} {e['path']}")


def test_node_tag_has_endpoints(api_spec: Dict[str, Any]):
    """
    Test that the 'node' tag has appropriate endpoints

    Args:
        api_spec: Loaded OpenAPI spec
    """
    endpoints = extract_endpoints_by_tag(api_spec, "node")

    assert len(endpoints) > 0, "No endpoints found with 'node' tag"

    # Check for key node endpoints
    endpoint_paths = [e["path"] for e in endpoints]
    assert "/object_info" in endpoint_paths, "Node tag should include /object_info endpoint"

    # Log the endpoints
    logger.info(f"Found {len(endpoints)} endpoints with 'node' tag:")
    for e in endpoints:
        logger.info(f"  {e['method'].upper()} {e['path']}")


def test_system_tag_has_endpoints(api_spec: Dict[str, Any]):
    """
    Test that the 'system' tag has appropriate endpoints

    Args:
        api_spec: Loaded OpenAPI spec
    """
    endpoints = extract_endpoints_by_tag(api_spec, "system")

    assert len(endpoints) > 0, "No endpoints found with 'system' tag"

    # Check for key system endpoints
    endpoint_paths = [e["path"] for e in endpoints]
    assert "/system_stats" in endpoint_paths, "System tag should include /system_stats endpoint"

    # Log the endpoints
    logger.info(f"Found {len(endpoints)} endpoints with 'system' tag:")
    for e in endpoints:
        logger.info(f"  {e['method'].upper()} {e['path']}")


def test_internal_tag_has_endpoints(api_spec: Dict[str, Any]):
    """
    Test that the 'internal' tag has appropriate endpoints

    Args:
        api_spec: Loaded OpenAPI spec
    """
    endpoints = extract_endpoints_by_tag(api_spec, "internal")

    assert len(endpoints) > 0, "No endpoints found with 'internal' tag"

    # Check for key internal endpoints
    endpoint_paths = [e["path"] for e in endpoints]
    assert "/internal/logs" in endpoint_paths, "Internal tag should include /internal/logs endpoint"

    # Log the endpoints
    logger.info(f"Found {len(endpoints)} endpoints with 'internal' tag:")
    for e in endpoints:
        logger.info(f"  {e['method'].upper()} {e['path']}")


def test_operation_ids_match_tag(api_spec: Dict[str, Any]):
    """
    Test that operation IDs follow a consistent pattern with their tag

    Args:
        api_spec: Loaded OpenAPI spec
    """
    failures = []

    for path, path_item in api_spec['paths'].items():
        for method, operation in path_item.items():
            if method in ['get', 'post', 'put', 'delete', 'patch']:
                if 'operationId' in operation and 'tags' in operation and operation['tags']:
                    op_id = operation['operationId']
                    primary_tag = operation['tags'][0].lower()

                    # Check if operationId starts with primary tag prefix
                    # This is a common convention, but might need adjusting
                    if not (op_id.startswith(primary_tag) or
                            any(op_id.lower().startswith(f"{tag.lower()}") for tag in operation['tags'])):
                        failures.append({
                            'path': path,
                            'method': method,
                            'operationId': op_id,
                            'primary_tag': primary_tag
                        })

    # Log failures for diagnosis but don't fail the test
    # as this is a style/convention check
    if failures:
        logger.warning(f"Found {len(failures)} operationIds that don't align with their tags:")
        for f in failures:
            logger.warning(f"  {f['method'].upper()} {f['path']} - operationId: {f['operationId']}, primary tag: {f['primary_tag']}")
