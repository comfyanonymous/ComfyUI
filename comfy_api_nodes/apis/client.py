import logging

"""
API Client Framework for api.comfy.org.

This module provides a flexible framework for making API requests from ComfyUI nodes.
It supports both synchronous and asynchronous API operations with proper type validation.

Key Components:
--------------
1. ApiClient - Handles HTTP requests with authentication and error handling
2. ApiEndpoint - Defines a single HTTP endpoint with its request/response models
3. ApiOperation - Executes a single synchronous API operation

Usage Examples:
--------------

# Example 1: Synchronous API Operation
# ------------------------------------
# For a simple API call that returns the result immediately:

# 1. Create the API client
api_client = ApiClient(
    base_url="https://api.example.com",
    api_key="your_api_key_here",
    timeout=30.0,
    verify_ssl=True
)

# 2. Define the endpoint
user_info_endpoint = ApiEndpoint(
    path="/v1/users/me",
    method=HttpMethod.GET,
    request_model=EmptyRequest,  # No request body needed
    response_model=UserProfile,   # Pydantic model for the response
    query_params=None
)

# 3. Create the request object
request = EmptyRequest()

# 4. Create and execute the operation
operation = ApiOperation(
    endpoint=user_info_endpoint,
    request=request
)
user_profile = operation.execute(client=api_client)  # Returns immediately with the result

"""

from typing import (
    Dict,
    Type,
    Optional,
    Any,
    TypeVar,
    Generic,
)
from pydantic import BaseModel
from enum import Enum
import json
import requests
from urllib.parse import urljoin

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R", bound=BaseModel)

class EmptyRequest(BaseModel):
    """Base class for empty request bodies.
    For GET requests, fields will be sent as query parameters."""

    pass


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class ApiClient:
    """
    Client for making HTTP requests to an API with authentication and error handling.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        verify_ssl: bool = True,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl

    def get_headers(self) -> Dict[str, str]:
        """Get headers for API requests, including authentication if available"""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path (will be joined with base_url)
            params: Query parameters
            json: JSON body data
            files: Files to upload
            headers: Additional headers

        Returns:
            Parsed JSON response

        Raises:
            requests.RequestException: If the request fails
        """
        url = urljoin(self.base_url, path)
        self.check_auth_token(self.api_key)
        # Combine default headers with any provided headers
        request_headers = self.get_headers()
        if headers:
            request_headers.update(headers)

        # Let requests handle the content type when files are present.
        if files:
            del request_headers["Content-Type"]

        logging.debug(f"[DEBUG] Request Headers: {request_headers}")
        logging.debug(f"[DEBUG] Files: {files}")
        logging.debug(f"[DEBUG] Params: {params}")
        logging.debug(f"[DEBUG] Json: {json}")

        try:
            # If files are present, use data parameter instead of json
            if files:
                form_data = {}
                if json:
                    form_data.update(json)
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    data=form_data,  # Use data instead of json
                    files=files,
                    headers=request_headers,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
            else:
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    headers=request_headers,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )

            # Raise exception for error status codes
            response.raise_for_status()
        except requests.ConnectionError:
            raise Exception(
                f"Unable to connect to the API server at {self.base_url}. Please check your internet connection or verify the service is available."
            )

        except requests.Timeout:
            raise Exception(
                f"Request timed out after {self.timeout} seconds. The server might be experiencing high load or the operation is taking longer than expected."
            )

        except requests.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, "response") else None
            error_message = f"HTTP Error: {str(e)}"

            # Try to extract detailed error message from JSON response
            try:
                if hasattr(e, "response") and e.response.content:
                    error_json = e.response.json()
                    if "error" in error_json and "message" in error_json["error"]:
                        error_message = f"API Error: {error_json['error']['message']}"
                        if "type" in error_json["error"]:
                            error_message += f" (Type: {error_json['error']['type']})"
                    else:
                        error_message = f"API Error: {error_json}"
            except Exception as json_error:
                # If we can't parse the JSON, fall back to the original error message
                logging.debug(f"[DEBUG] Failed to parse error response: {str(json_error)}")

            logging.debug(f"[DEBUG] API Error: {error_message} (Status: {status_code})")
            if hasattr(e, "response") and e.response.content:
                logging.debug(f"[DEBUG] Response content: {e.response.content}")
            if status_code == 401:
                error_message = "Unauthorized: Please login first to use this node."
            if status_code == 402:
                error_message = "Payment Required: Please add credits to your account to use this node."
            if status_code == 409:
                error_message = "There is a problem with your account. Please contact support@comfy.org. "
            if status_code == 429:
                error_message = "Rate Limit Exceeded: Please try again later."
            raise Exception(error_message)

        # Parse and return JSON response
        if response.content:
            return response.json()
        return {}

    def check_auth_token(self, auth_token):
        """Verify that an auth token is present."""
        if auth_token is None:
            raise Exception("Unauthorized: Please login first to use this node.")
        return auth_token


class ApiEndpoint(Generic[T, R]):
    """Defines an API endpoint with its request and response types"""

    def __init__(
        self,
        path: str,
        method: HttpMethod,
        request_model: Type[T],
        response_model: Type[R],
        query_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize an API endpoint definition.

        Args:
            path: The URL path for this endpoint, can include placeholders like {id}
            method: The HTTP method to use (GET, POST, etc.)
            request_model: Pydantic model class that defines the structure and validation rules for API requests to this endpoint
            response_model: Pydantic model class that defines the structure and validation rules for API responses from this endpoint
            query_params: Optional dictionary of query parameters to include in the request
        """
        self.path = path
        self.method = method
        self.request_model = request_model
        self.response_model = response_model
        self.query_params = query_params or {}


class SynchronousOperation(Generic[T, R]):
    """
    Represents a single synchronous API operation.
    """

    def __init__(
        self,
        endpoint: ApiEndpoint[T, R],
        request: T,
        files: Optional[Dict[str, Any]] = None,
        api_base: str = "https://api.comfy.org",
        auth_token: Optional[str] = None,
        timeout: float = 604800.0,
        verify_ssl: bool = True,
    ):
        self.endpoint = endpoint
        self.request = request
        self.response = None
        self.error = None
        self.api_base = api_base
        self.auth_token = auth_token
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.files = files
    def execute(self, client: Optional[ApiClient] = None) -> R:
        """Execute the API operation using the provided client or create one"""
        try:
            # Create client if not provided
            if client is None:
                if self.api_base is None:
                    raise ValueError("Either client or api_base must be provided")
                client = ApiClient(
                    base_url=self.api_base,
                    api_key=self.auth_token,
                    timeout=self.timeout,
                    verify_ssl=self.verify_ssl,
                )

            # Convert request model to dict, but use None for EmptyRequest
            request_dict = None if isinstance(self.request, EmptyRequest) else self.request.model_dump(exclude_none=True)

            # Debug log for request
            logging.debug(f"[DEBUG] API Request: {self.endpoint.method.value} {self.endpoint.path}")
            logging.debug(f"[DEBUG] Request Data: {json.dumps(request_dict, indent=2)}")
            logging.debug(f"[DEBUG] Query Params: {self.endpoint.query_params}")

            # Make the request
            resp = client.request(
                method=self.endpoint.method.value,
                path=self.endpoint.path,
                json=request_dict,
                params=self.endpoint.query_params,
                files=self.files,
            )

            # Debug log for response
            logging.debug("=" * 50)
            logging.debug("[DEBUG] RESPONSE DETAILS:")
            logging.debug("[DEBUG] Status Code: 200 (Success)")
            logging.debug(f"[DEBUG] Response Body: {json.dumps(resp, indent=2)}")
            logging.debug("=" * 50)

            # Parse and return the response
            return self._parse_response(resp)

        except Exception as e:
            logging.debug(f"[DEBUG] API Exception: {str(e)}")
            raise Exception(str(e))

    def _parse_response(self, resp):
        """Parse response data - can be overridden by subclasses"""
        # The response is already the complete object, don't extract just the "data" field
        # as that would lose the outer structure (created timestamp, etc.)

        # Parse response using the provided model
        self.response = self.endpoint.response_model.model_validate(resp)
        logging.debug(f"[DEBUG] Parsed Response: {self.response}")
        return self.response
