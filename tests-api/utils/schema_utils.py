"""
Utilities for working with OpenAPI schemas
"""
from typing import Any, Dict, List, Optional, Set, Tuple


def extract_required_parameters(
    spec: Dict[str, Any],
    path: str,
    method: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract required parameters for a specific endpoint

    Args:
        spec: Parsed OpenAPI specification
        path: API path (e.g., '/prompt')
        method: HTTP method (e.g., 'get', 'post')

    Returns:
        Tuple of (path_params, query_params) containing required parameters
    """
    method = method.lower()
    path_params = []
    query_params = []

    # Handle path not found
    if path not in spec['paths']:
        return path_params, query_params

    # Handle method not found
    if method not in spec['paths'][path]:
        return path_params, query_params

    # Get parameters
    params = spec['paths'][path][method].get('parameters', [])

    for param in params:
        if param.get('required', False):
            if param.get('in') == 'path':
                path_params.append(param)
            elif param.get('in') == 'query':
                query_params.append(param)

    return path_params, query_params


def get_request_body_schema(
    spec: Dict[str, Any],
    path: str,
    method: str
) -> Optional[Dict[str, Any]]:
    """
    Get request body schema for a specific endpoint

    Args:
        spec: Parsed OpenAPI specification
        path: API path (e.g., '/prompt')
        method: HTTP method (e.g., 'get', 'post')

    Returns:
        Request body schema or None if not found
    """
    method = method.lower()

    # Handle path not found
    if path not in spec['paths']:
        return None

    # Handle method not found
    if method not in spec['paths'][path]:
        return None

    # Handle no request body
    request_body = spec['paths'][path][method].get('requestBody', {})
    if not request_body or 'content' not in request_body:
        return None

    # Get schema from first content type
    content_types = request_body['content']
    first_content_type = next(iter(content_types))

    if 'schema' not in content_types[first_content_type]:
        return None

    return content_types[first_content_type]['schema']


def extract_endpoints_by_tag(spec: Dict[str, Any], tag: str) -> List[Dict[str, Any]]:
    """
    Extract all endpoints with a specific tag

    Args:
        spec: Parsed OpenAPI specification
        tag: Tag to filter by

    Returns:
        List of endpoint details
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


def get_all_tags(spec: Dict[str, Any]) -> Set[str]:
    """
    Get all tags used in the API spec

    Args:
        spec: Parsed OpenAPI specification

    Returns:
        Set of tag names
    """
    tags = set()

    for path_item in spec['paths'].values():
        for operation in path_item.values():
            if isinstance(operation, dict) and 'tags' in operation:
                tags.update(operation['tags'])

    return tags


def get_schema_examples(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all examples from component schemas

    Args:
        spec: Parsed OpenAPI specification

    Returns:
        Dict mapping schema names to examples
    """
    examples = {}

    if 'components' not in spec or 'schemas' not in spec['components']:
        return examples

    for name, schema in spec['components']['schemas'].items():
        if 'example' in schema:
            examples[name] = schema['example']

    return examples
