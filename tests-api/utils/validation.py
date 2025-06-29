"""
Utilities for API response validation against OpenAPI spec
"""
import yaml
import jsonschema
from typing import Any, Dict, List, Optional, Union


def load_openapi_spec(spec_path: str) -> Dict[str, Any]:
    """
    Load the OpenAPI specification from a YAML file

    Args:
        spec_path: Path to the OpenAPI specification file

    Returns:
        Dict containing the parsed OpenAPI spec
    """
    with open(spec_path, 'r') as f:
        return yaml.safe_load(f)


def get_endpoint_schema(
    spec: Dict[str, Any],
    path: str,
    method: str,
    status_code: str = '200'
) -> Optional[Dict[str, Any]]:
    """
    Extract response schema for a specific endpoint from OpenAPI spec

    Args:
        spec: Parsed OpenAPI specification
        path: API path (e.g., '/prompt')
        method: HTTP method (e.g., 'get', 'post')
        status_code: HTTP status code to get schema for

    Returns:
        Schema dict or None if not found
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


def resolve_schema_refs(schema: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve $ref references in a schema and convert OpenAPI nullable to JSON Schema

    Args:
        schema: Schema that may contain references
        spec: Full OpenAPI spec with component definitions

    Returns:
        Schema with references resolved
    """
    if not isinstance(schema, dict):
        return schema

    result = {}

    # Check if this schema has nullable: true with a type
    if schema.get('nullable') is True and 'type' in schema:
        # Convert OpenAPI nullable syntax to JSON Schema oneOf
        original_type = schema['type']
        result['oneOf'] = [
            {'type': original_type},
            {'type': 'null'}
        ]
        # Copy other properties except nullable and type
        for key, value in schema.items():
            if key not in ['nullable', 'type']:
                if isinstance(value, dict):
                    result[key] = resolve_schema_refs(value, spec)
                elif isinstance(value, list):
                    result[key] = [
                        resolve_schema_refs(item, spec) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    result[key] = value
    else:
        # Normal processing
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
                # Pass through other values (skip nullable as it's OpenAPI specific)
                if key != 'nullable':
                    result[key] = value

    return result


def validate_response(
    response_data: Union[Dict[str, Any], List[Any]],
    spec: Dict[str, Any],
    path: str,
    method: str,
    status_code: str = '200'
) -> Dict[str, Any]:
    """
    Validate a response against the OpenAPI schema

    Args:
        response_data: Response data to validate
        spec: Parsed OpenAPI specification
        path: API path (e.g., '/prompt')
        method: HTTP method (e.g., 'get', 'post')
        status_code: HTTP status code to validate against

    Returns:
        Dict with validation result containing:
            - valid: bool indicating if validation passed
            - errors: List of validation errors if any
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
        jsonschema.validate(instance=response_data, schema=resolved_schema)
        return {'valid': True, 'errors': []}
    except jsonschema.exceptions.ValidationError as e:
        return {'valid': False, 'errors': [str(e)]}


def get_all_endpoints(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
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
