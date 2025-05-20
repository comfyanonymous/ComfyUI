# ComfyUI API Testing

This directory contains tests for validating the ComfyUI OpenAPI specification against a running instance of ComfyUI.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure you have a running instance of ComfyUI (default: http://127.0.0.1:8188)

## Running the Tests

Run all tests with pytest:

```bash
cd tests-api
pytest
```

Run specific test files:

```bash
pytest test_spec_validation.py
pytest test_endpoint_existence.py
pytest test_schema_validation.py
pytest test_api_by_tag.py
```

Run tests with more verbose output:

```bash
pytest -v
```

## Test Categories

The tests are organized into several categories:

1. **Spec Validation**: Validates that the OpenAPI specification is valid.
2. **Endpoint Existence**: Tests that the endpoints defined in the spec exist on the server.
3. **Schema Validation**: Tests that the server responses match the schemas defined in the spec.
4. **Tag-Based Tests**: Tests that the API's tag organization is consistent.

## Using a Different Server

By default, the tests connect to `http://127.0.0.1:8188`. To test against a different server, set the `COMFYUI_SERVER_URL` environment variable:

```bash
COMFYUI_SERVER_URL=http://example.com:8188 pytest
```

## Test Structure

- `conftest.py`: Contains pytest fixtures used by the tests.
- `utils/`: Contains utility functions for working with the OpenAPI spec.
- `test_*.py`: The actual test files.
- `resources/`: Contains resources used by the tests (e.g., sample workflows).

## Extending the Tests

To add new tests:

1. For testing new endpoints, add them to the appropriate test file based on their category.
2. For testing more complex functionality, create a new test file following the established patterns.

## Notes

- Tests that require a running server will be skipped if the server is not available.
- Some tests may fail if the server doesn't match the specification exactly.
- The tests don't modify any data on the server (they're read-only).