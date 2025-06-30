"""
Tests for validating the OpenAPI specification
"""
import pytest
from openapi_spec_validator import validate_spec
from openapi_spec_validator.exceptions import OpenAPISpecValidatorError
from typing import Dict, Any


def test_openapi_spec_is_valid(api_spec: Dict[str, Any]):
    """
    Test that the OpenAPI specification is valid

    Args:
        api_spec: Loaded OpenAPI spec
    """
    try:
        validate_spec(api_spec)
    except OpenAPISpecValidatorError as e:
        pytest.fail(f"OpenAPI spec validation failed: {str(e)}")


def test_spec_has_info(api_spec: Dict[str, Any]):
    """
    Test that the OpenAPI spec has the required info section

    Args:
        api_spec: Loaded OpenAPI spec
    """
    assert 'info' in api_spec, "Spec must have info section"
    assert 'title' in api_spec['info'], "Info must have title"
    assert 'version' in api_spec['info'], "Info must have version"


def test_spec_has_paths(api_spec: Dict[str, Any]):
    """
    Test that the OpenAPI spec has paths defined

    Args:
        api_spec: Loaded OpenAPI spec
    """
    assert 'paths' in api_spec, "Spec must have paths section"
    assert len(api_spec['paths']) > 0, "Spec must have at least one path"


def test_spec_has_components(api_spec: Dict[str, Any]):
    """
    Test that the OpenAPI spec has components defined

    Args:
        api_spec: Loaded OpenAPI spec
    """
    assert 'components' in api_spec, "Spec must have components section"
    assert 'schemas' in api_spec['components'], "Components must have schemas"


def test_workflow_endpoints_exist(api_spec: Dict[str, Any]):
    """
    Test that core workflow endpoints are defined

    Args:
        api_spec: Loaded OpenAPI spec
    """
    assert '/api/prompt' in api_spec['paths'], "Spec must define /api/prompt endpoint"
    assert 'post' in api_spec['paths']['/api/prompt'], "Spec must define POST /api/prompt"
    assert 'get' in api_spec['paths']['/api/prompt'], "Spec must define GET /api/prompt"


def test_image_endpoints_exist(api_spec: Dict[str, Any]):
    """
    Test that core image endpoints are defined

    Args:
        api_spec: Loaded OpenAPI spec
    """
    assert '/api/upload/image' in api_spec['paths'], "Spec must define /api/upload/image endpoint"
    assert '/api/view' in api_spec['paths'], "Spec must define /api/view endpoint"


def test_model_endpoints_exist(api_spec: Dict[str, Any]):
    """
    Test that core model endpoints are defined

    Args:
        api_spec: Loaded OpenAPI spec
    """
    assert '/api/models' in api_spec['paths'], "Spec must define /api/models endpoint"
    assert '/api/models/{folder}' in api_spec['paths'], "Spec must define /api/models/{folder} endpoint"


def test_operation_ids_are_unique(api_spec: Dict[str, Any]):
    """
    Test that all operationIds are unique

    Args:
        api_spec: Loaded OpenAPI spec
    """
    operation_ids = []

    for path, path_item in api_spec['paths'].items():
        for method, operation in path_item.items():
            if method in ['get', 'post', 'put', 'delete', 'patch']:
                if 'operationId' in operation:
                    operation_ids.append(operation['operationId'])

    # Check for duplicates
    duplicates = set([op_id for op_id in operation_ids if operation_ids.count(op_id) > 1])
    assert len(duplicates) == 0, f"Found duplicate operationIds: {duplicates}"


def test_all_endpoints_have_operation_ids(api_spec: Dict[str, Any]):
    """
    Test that all endpoints have operationIds

    Args:
        api_spec: Loaded OpenAPI spec
    """
    missing = []

    for path, path_item in api_spec['paths'].items():
        for method, operation in path_item.items():
            if method in ['get', 'post', 'put', 'delete', 'patch']:
                if 'operationId' not in operation:
                    missing.append(f"{method.upper()} {path}")

    assert len(missing) == 0, f"Found endpoints without operationIds: {missing}"


def test_all_endpoints_have_tags(api_spec: Dict[str, Any]):
    """
    Test that all endpoints have tags

    Args:
        api_spec: Loaded OpenAPI spec
    """
    missing = []

    for path, path_item in api_spec['paths'].items():
        for method, operation in path_item.items():
            if method in ['get', 'post', 'put', 'delete', 'patch']:
                if 'tags' not in operation or not operation['tags']:
                    missing.append(f"{method.upper()} {path}")

    assert len(missing) == 0, f"Found endpoints without tags: {missing}"
