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
    auth_token="your_auth_token_here",
    comfy_api_key="your_comfy_api_key_here",
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


# Example 2: Asynchronous API Operation with Polling
# -------------------------------------------------
# For an API that starts a task and requires polling for completion:

# 1. Define the endpoints (initial request and polling)
generate_image_endpoint = ApiEndpoint(
    path="/v1/images/generate",
    method=HttpMethod.POST,
    request_model=ImageGenerationRequest,
    response_model=TaskCreatedResponse,
    query_params=None
)

check_task_endpoint = ApiEndpoint(
    path="/v1/tasks/{task_id}",
    method=HttpMethod.GET,
    request_model=EmptyRequest,
    response_model=ImageGenerationResult,
    query_params=None
)

# 2. Create the request object
request = ImageGenerationRequest(
    prompt="a beautiful sunset over mountains",
    width=1024,
    height=1024,
    num_images=1
)

# 3. Create and execute the polling operation
operation = PollingOperation(
    initial_endpoint=generate_image_endpoint,
    initial_request=request,
    poll_endpoint=check_task_endpoint,
    task_id_field="task_id",
    status_field="status",
    completed_statuses=["completed"],
    failed_statuses=["failed", "error"]
)

# This will make the initial request and then poll until completion
result = operation.execute(client=api_client)  # Returns the final ImageGenerationResult when done
"""

from __future__ import annotations
import logging
import time
import io
import socket
from typing import Dict, Type, Optional, Any, TypeVar, Generic, Callable, Tuple
from enum import Enum
import json
import requests
from urllib.parse import urljoin, urlparse
from pydantic import BaseModel, Field
import uuid # For generating unique operation IDs

from server import PromptServer
from comfy.cli_args import args
from comfy import utils
from . import request_logger

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R", bound=BaseModel)
P = TypeVar("P", bound=BaseModel)  # For poll response

PROGRESS_BAR_MAX = 100


class NetworkError(Exception):
    """Base exception for network-related errors with diagnostic information."""
    pass


class LocalNetworkError(NetworkError):
    """Exception raised when local network connectivity issues are detected."""
    pass


class ApiServerError(NetworkError):
    """Exception raised when the API server is unreachable but internet is working."""
    pass


class EmptyRequest(BaseModel):
    """Base class for empty request bodies.
    For GET requests, fields will be sent as query parameters."""

    pass


class UploadRequest(BaseModel):
    file_name: str = Field(..., description="Filename to upload")
    content_type: Optional[str] = Field(
        None,
        description="Mime type of the file. For example: image/png, image/jpeg, video/mp4, etc.",
    )


class UploadResponse(BaseModel):
    download_url: str = Field(..., description="URL to GET uploaded file")
    upload_url: str = Field(..., description="URL to PUT file to upload")


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class ApiClient:
    """
    Client for making HTTP requests to an API with authentication, error handling, and retry logic.
    """

    def __init__(
        self,
        base_url: str,
        auth_token: Optional[str] = None,
        comfy_api_key: Optional[str] = None,
        timeout: float = 3600.0,
        verify_ssl: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff_factor: float = 2.0,
        retry_status_codes: Optional[Tuple[int, ...]] = None,
    ):
        self.base_url = base_url
        self.auth_token = auth_token
        self.comfy_api_key = comfy_api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff_factor = retry_backoff_factor
        # Default retry status codes: 408 (Request Timeout), 429 (Too Many Requests),
        # 500, 502, 503, 504 (Server Errors)
        self.retry_status_codes = retry_status_codes or (408, 429, 500, 502, 503, 504)

    def _generate_operation_id(self, path: str) -> str:
        """Generates a unique operation ID for logging."""
        return f"{path.strip('/').replace('/', '_')}_{uuid.uuid4().hex[:8]}"

    def _create_json_payload_args(
        self,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        return {
            "json": data,
            "headers": headers,
        }

    def _create_form_data_args(
        self,
        data: Dict[str, Any],
        files: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        multipart_parser = None,
    ) -> Dict[str, Any]:
        if headers and "Content-Type" in headers:
            del headers["Content-Type"]

        if multipart_parser:
            data = multipart_parser(data)

        return {
            "data": data,
            "files": files,
            "headers": headers,
        }

    def _create_urlencoded_form_data_args(
        self,
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        headers = headers or {}
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        return {
            "data": data,
            "headers": headers,
        }

    def get_headers(self) -> Dict[str, str]:
        """Get headers for API requests, including authentication if available"""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        elif self.comfy_api_key:
            headers["X-API-KEY"] = self.comfy_api_key

        return headers

    def _check_connectivity(self, target_url: str) -> Dict[str, bool]:
        """
        Check connectivity to determine if network issues are local or server-related.

        Args:
            target_url: URL to check connectivity to

        Returns:
            Dictionary with connectivity status details
        """
        results = {
            "internet_accessible": False,
            "api_accessible": False,
            "is_local_issue": False,
            "is_api_issue": False
        }

        # First check basic internet connectivity using a reliable external site
        try:
            # Use a reliable external domain for checking basic connectivity
            check_response = requests.get("https://www.google.com",
                                         timeout=5.0,
                                         verify=self.verify_ssl)
            if check_response.status_code < 500:
                results["internet_accessible"] = True
        except (requests.RequestException, socket.error):
            results["internet_accessible"] = False
            results["is_local_issue"] = True
            return results

        # Now check API server connectivity
        try:
            # Extract domain from the target URL to do a simpler health check
            parsed_url = urlparse(target_url)
            api_base = f"{parsed_url.scheme}://{parsed_url.netloc}"

            # Try to reach the API domain
            api_response = requests.get(f"{api_base}/health", timeout=5.0, verify=self.verify_ssl)
            if api_response.status_code < 500:
                results["api_accessible"] = True
            else:
                results["api_accessible"] = False
                results["is_api_issue"] = True
        except requests.RequestException:
            results["api_accessible"] = False
            # If we can reach the internet but not the API, it's an API issue
            results["is_api_issue"] = True

        return results

    def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        content_type: str = "application/json",
        multipart_parser: Callable = None,
        retry_count: int = 0,  # Used internally for tracking retries
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API with automatic retries for transient errors.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path (will be joined with base_url)
            params: Query parameters
            data: body data
            files: Files to upload
            headers: Additional headers
            content_type: Content type of the request. Defaults to application/json.
            retry_count: Internal parameter for tracking retries, do not set manually

        Returns:
            Parsed JSON response

        Raises:
            LocalNetworkError: If local network connectivity issues are detected
            ApiServerError: If the API server is unreachable but internet is working
            Exception: For other request failures
        """
        url = urljoin(self.base_url, path)
        self.check_auth(self.auth_token, self.comfy_api_key)
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
        logging.debug(f"[DEBUG] Data: {data}")

        if content_type == "application/x-www-form-urlencoded":
            payload_args = self._create_urlencoded_form_data_args(data, request_headers)
        elif content_type == "multipart/form-data":
            payload_args = self._create_form_data_args(
                data, files, request_headers, multipart_parser
            )
        else:
            payload_args = self._create_json_payload_args(data, request_headers)

        operation_id = self._generate_operation_id(path)
        request_logger.log_request_response(
            operation_id=operation_id,
            request_method=method,
            request_url=url,
            request_headers=request_headers,
            request_params=params,
            request_data=data if content_type == "application/json" else "[form-data or other]"
        )

        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                timeout=self.timeout,
                verify=self.verify_ssl,
                **payload_args,
            )

            # Check if we should retry based on status code
            if (response.status_code in self.retry_status_codes and
                retry_count < self.max_retries):

                # Calculate delay with exponential backoff
                delay = self.retry_delay * (self.retry_backoff_factor ** retry_count)

                logging.warning(
                    f"Request failed with status {response.status_code}. "
                    f"Retrying in {delay:.2f}s ({retry_count + 1}/{self.max_retries})"
                )

                time.sleep(delay)
                return self.request(
                    method=method,
                    path=path,
                    params=params,
                    data=data,
                    files=files,
                    headers=headers,
                    content_type=content_type,
                    multipart_parser=multipart_parser,
                    retry_count=retry_count + 1,
                )

            # Raise exception for error status codes
            response.raise_for_status()

            # Log successful response
            response_content_to_log = response.content
            try:
                # Attempt to parse JSON for prettier logging, fallback to raw content
                response_content_to_log = response.json()
            except json.JSONDecodeError:
                pass # Keep as bytes/str if not JSON

            request_logger.log_request_response(
                operation_id=operation_id,
                request_method=method, # Pass request details again for context in log
                request_url=url,
                response_status_code=response.status_code,
                response_headers=dict(response.headers),
                response_content=response_content_to_log
            )

        except requests.ConnectionError as e:
            error_message = f"ConnectionError: {str(e)}"
            request_logger.log_request_response(
                operation_id=operation_id,
                request_method=method,
                request_url=url,
                error_message=error_message
            )
            # Only perform connectivity check if we've exhausted all retries
            if retry_count >= self.max_retries:
                # Check connectivity to determine if it's a local or API issue
                connectivity = self._check_connectivity(self.base_url)

                if connectivity["is_local_issue"]:
                    raise LocalNetworkError(
                        "Unable to connect to the API server due to local network issues. "
                        "Please check your internet connection and try again."
                    ) from e
                elif connectivity["is_api_issue"]:
                    raise ApiServerError(
                        f"The API server at {self.base_url} is currently unreachable. "
                        f"The service may be experiencing issues. Please try again later."
                    ) from e

            # If we haven't exhausted retries yet, retry the request
            if retry_count < self.max_retries:
                delay = self.retry_delay * (self.retry_backoff_factor ** retry_count)
                logging.warning(
                    f"Connection error: {str(e)}. "
                    f"Retrying in {delay:.2f}s ({retry_count + 1}/{self.max_retries})"
                )
                time.sleep(delay)
                return self.request(
                    method=method,
                    path=path,
                    params=params,
                    data=data,
                    files=files,
                    headers=headers,
                    content_type=content_type,
                    multipart_parser=multipart_parser,
                    retry_count=retry_count + 1,
                )

            # If we've exhausted retries and didn't identify the specific issue,
            # raise a generic exception
            final_error_message = (
                f"Unable to connect to the API server after {self.max_retries} attempts. "
                f"Please check your internet connection or try again later."
            )
            request_logger.log_request_response( # Log final failure
                operation_id=operation_id,
                request_method=method, request_url=url,
                error_message=final_error_message
            )
            raise Exception(final_error_message) from e

        except requests.Timeout as e:
            error_message = f"Timeout: {str(e)}"
            request_logger.log_request_response(
                operation_id=operation_id,
                request_method=method, request_url=url,
                error_message=error_message
            )
            # Retry timeouts if we haven't exhausted retries
            if retry_count < self.max_retries:
                delay = self.retry_delay * (self.retry_backoff_factor ** retry_count)
                logging.warning(
                    f"Request timed out. "
                    f"Retrying in {delay:.2f}s ({retry_count + 1}/{self.max_retries})"
                )
                time.sleep(delay)
                return self.request(
                    method=method,
                    path=path,
                    params=params,
                    data=data,
                    files=files,
                    headers=headers,
                    content_type=content_type,
                    multipart_parser=multipart_parser,
                    retry_count=retry_count + 1,
                )
            final_error_message = (
                f"Request timed out after {self.timeout} seconds and {self.max_retries} retry attempts. "
                f"The server might be experiencing high load or the operation is taking longer than expected."
            )
            request_logger.log_request_response( # Log final failure
                operation_id=operation_id,
                request_method=method, request_url=url,
                error_message=final_error_message
            )
            raise Exception(final_error_message) from e

        except requests.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, "response") else None
            original_error_message = f"HTTP Error: {str(e)}"
            error_content_for_log = None
            if hasattr(e, "response") and e.response is not None:
                error_content_for_log = e.response.content
                try:
                    error_content_for_log = e.response.json()
                except json.JSONDecodeError:
                    pass


            # Try to extract detailed error message from JSON response for user display
            # but log the full error content.
            user_display_error_message = original_error_message

            try:
                if hasattr(e, "response") and e.response is not None and e.response.content:
                    error_json = e.response.json()
                    if "error" in error_json and "message" in error_json["error"]:
                        user_display_error_message = f"API Error: {error_json['error']['message']}"
                        if "type" in error_json["error"]:
                            user_display_error_message += f" (Type: {error_json['error']['type']})"
                    elif isinstance(error_json, dict): # Handle cases where error is just a JSON dict
                        user_display_error_message = f"API Error: {json.dumps(error_json)}"
                    else: # Non-dict JSON error
                        user_display_error_message = f"API Error: {str(error_json)}"
            except json.JSONDecodeError:
                # If not JSON, use the raw content if it's not too long, or a summary
                if hasattr(e, "response") and e.response is not None and e.response.content:
                    raw_content = e.response.content.decode(errors='ignore')
                    if len(raw_content) < 200: # Arbitrary limit for display
                        user_display_error_message = f"API Error (raw): {raw_content}"
                    else:
                        user_display_error_message = f"API Error (raw, status {status_code})"

            request_logger.log_request_response(
                operation_id=operation_id,
                request_method=method, request_url=url,
                response_status_code=status_code,
                response_headers=dict(e.response.headers) if hasattr(e, "response") and e.response is not None else None,
                response_content=error_content_for_log,
                error_message=original_error_message # Log the original exception string as error
            )

            logging.debug(f"[DEBUG] API Error: {user_display_error_message} (Status: {status_code})")
            if hasattr(e, "response") and e.response is not None and e.response.content:
                logging.debug(f"[DEBUG] Response content: {e.response.content}")

            # Retry if the status code is in our retry list and we haven't exhausted retries
            if (status_code in self.retry_status_codes and
                retry_count < self.max_retries):

                delay = self.retry_delay * (self.retry_backoff_factor ** retry_count)
                logging.warning(
                    f"HTTP error {status_code}. "
                    f"Retrying in {delay:.2f}s ({retry_count + 1}/{self.max_retries})"
                )
                time.sleep(delay)
                return self.request(
                    method=method,
                    path=path,
                    params=params,
                    data=data,
                    files=files,
                    headers=headers,
                    content_type=content_type,
                    multipart_parser=multipart_parser,
                    retry_count=retry_count + 1,
                )

            # Specific error messages for common status codes for user display
            if status_code == 401:
                user_display_error_message = "Unauthorized: Please login first to use this node."
            elif status_code == 402:
                user_display_error_message = "Payment Required: Please add credits to your account to use this node."
            elif status_code == 409:
                user_display_error_message = "There is a problem with your account. Please contact support@comfy.org."
            elif status_code == 429:
                user_display_error_message = "Rate Limit Exceeded: Please try again later."
            # else, user_display_error_message remains as parsed from response or original HTTPError string

            raise Exception(user_display_error_message) # Raise with the user-friendly message

        # Parse and return JSON response
        if response.content:
            return response.json()
        return {}

    def check_auth(self, auth_token, comfy_api_key):
        """Verify that an auth token is present or comfy_api_key is present"""
        if auth_token is None and comfy_api_key is None:
            raise Exception("Unauthorized: Please login first to use this node.")
        return auth_token or comfy_api_key

    @staticmethod
    def upload_file(
        upload_url: str,
        file: io.BytesIO | str,
        content_type: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff_factor: float = 2.0,
    ):
        """Upload a file to the API with retry logic.

        Args:
            upload_url: The URL to upload to
            file: Either a file path string, BytesIO object, or tuple of (file_path, filename)
            content_type: Optional mime type to set for the upload
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            retry_backoff_factor: Multiplier for the delay after each retry
        """
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type

        # Prepare the file data
        if isinstance(file, io.BytesIO):
            file.seek(0)  # Ensure we're at the start of the file
            data = file.read()
        elif isinstance(file, str):
            with open(file, "rb") as f:
                data = f.read()
        else:
            raise ValueError("File must be either a BytesIO object or a file path string")

        # Try the upload with retries
        last_exception = None
        operation_id = f"upload_{upload_url.split('/')[-1]}_{uuid.uuid4().hex[:8]}" # Simplified ID for uploads

        # Log initial attempt (without full file data for brevity)
        request_logger.log_request_response(
            operation_id=operation_id,
            request_method="PUT",
            request_url=upload_url,
            request_headers=headers,
            request_data=f"[File data of type {content_type or 'unknown'}, size {len(data)} bytes]"
        )

        for retry_attempt in range(max_retries + 1):
            try:
                response = requests.put(upload_url, data=data, headers=headers)
                response.raise_for_status()
                request_logger.log_request_response(
                    operation_id=operation_id,
                    request_method="PUT", request_url=upload_url, # For context
                    response_status_code=response.status_code,
                    response_headers=dict(response.headers),
                    response_content="File uploaded successfully." # Or response.text if available
                )
                return response

            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                last_exception = e
                error_message_for_log = f"{type(e).__name__}: {str(e)}"
                response_content_for_log = None
                status_code_for_log = None
                headers_for_log = None

                if hasattr(e, 'response') and e.response is not None:
                    status_code_for_log = e.response.status_code
                    headers_for_log = dict(e.response.headers)
                    try:
                        response_content_for_log = e.response.json()
                    except json.JSONDecodeError:
                        response_content_for_log = e.response.content


                request_logger.log_request_response(
                    operation_id=operation_id,
                    request_method="PUT", request_url=upload_url,
                    response_status_code=status_code_for_log,
                    response_headers=headers_for_log,
                    response_content=response_content_for_log,
                    error_message=error_message_for_log
                )

                if retry_attempt < max_retries:
                    delay = retry_delay * (retry_backoff_factor ** retry_attempt)
                    logging.warning(
                        f"File upload failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s ({retry_attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    break # Max retries reached

        # If we've exhausted all retries, determine the final error type and raise
        final_error_message = f"Failed to upload file after {max_retries + 1} attempts. Error: {str(last_exception)}"
        try:
            # Check basic internet connectivity
            check_response = requests.get("https://www.google.com", timeout=5.0, verify=True) # Assuming verify=True is desired
            if check_response.status_code >= 500: # Google itself has an issue (rare)
                 final_error_message = (f"Failed to upload file. Internet connectivity check to Google failed "
                                       f"(status {check_response.status_code}). Original error: {str(last_exception)}")
                 # Not raising LocalNetworkError here as Google itself might be down.
            # If Google is reachable, the issue is likely with the upload server or a more specific local problem
            # not caught by a simple Google ping (e.g., DNS for the specific upload URL, firewall).
            # The original last_exception is probably most relevant.

        except (requests.RequestException, socket.error) as conn_check_exc:
            # Could not reach Google, likely a local network issue
            final_error_message = (f"Failed to upload file due to network connectivity issues "
                                   f"(cannot reach Google: {str(conn_check_exc)}). "
                                   f"Original upload error: {str(last_exception)}")
            request_logger.log_request_response( # Log final failure reason
                operation_id=operation_id,
                request_method="PUT", request_url=upload_url,
                error_message=final_error_message
            )
            raise LocalNetworkError(final_error_message) from last_exception

        request_logger.log_request_response( # Log final failure reason if not LocalNetworkError
            operation_id=operation_id,
            request_method="PUT", request_url=upload_url,
            error_message=final_error_message
        )
        raise Exception(final_error_message) from last_exception


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
        api_base: str | None = None,
        auth_token: Optional[str] = None,
        comfy_api_key: Optional[str] = None,
        auth_kwargs: Optional[Dict[str,str]] = None,
        timeout: float = 604800.0,
        verify_ssl: bool = True,
        content_type: str = "application/json",
        multipart_parser: Callable = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff_factor: float = 2.0,
    ):
        self.endpoint = endpoint
        self.request = request
        self.response = None
        self.error = None
        self.api_base: str = api_base or args.comfy_api_base
        self.auth_token = auth_token
        self.comfy_api_key = comfy_api_key
        if auth_kwargs is not None:
            self.auth_token = auth_kwargs.get("auth_token", self.auth_token)
            self.comfy_api_key = auth_kwargs.get("comfy_api_key", self.comfy_api_key)
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.files = files
        self.content_type = content_type
        self.multipart_parser = multipart_parser
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff_factor = retry_backoff_factor

    def execute(self, client: Optional[ApiClient] = None) -> R:
        """Execute the API operation using the provided client or create one with retry support"""
        try:
            # Create client if not provided
            if client is None:
                client = ApiClient(
                    base_url=self.api_base,
                    auth_token=self.auth_token,
                    comfy_api_key=self.comfy_api_key,
                    timeout=self.timeout,
                    verify_ssl=self.verify_ssl,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay,
                    retry_backoff_factor=self.retry_backoff_factor,
                )

            # Convert request model to dict, but use None for EmptyRequest
            request_dict = (
                None
                if isinstance(self.request, EmptyRequest)
                else self.request.model_dump(exclude_none=True)
            )
            if request_dict:
                for key, value in request_dict.items():
                    if isinstance(value, Enum):
                        request_dict[key] = value.value

            # Debug log for request
            logging.debug(
                f"[DEBUG] API Request: {self.endpoint.method.value} {self.endpoint.path}"
            )
            logging.debug(f"[DEBUG] Request Data: {json.dumps(request_dict, indent=2)}")
            logging.debug(f"[DEBUG] Query Params: {self.endpoint.query_params}")

            # Make the request with built-in retry
            resp = client.request(
                method=self.endpoint.method.value,
                path=self.endpoint.path,
                data=request_dict,
                params=self.endpoint.query_params,
                files=self.files,
                content_type=self.content_type,
                multipart_parser=self.multipart_parser
            )

            # Debug log for response
            logging.debug("=" * 50)
            logging.debug("[DEBUG] RESPONSE DETAILS:")
            logging.debug("[DEBUG] Status Code: 200 (Success)")
            logging.debug(f"[DEBUG] Response Body: {json.dumps(resp, indent=2)}")
            logging.debug("=" * 50)

            # Parse and return the response
            return self._parse_response(resp)

        except LocalNetworkError as e:
            # Propagate specific network error types
            logging.error(f"[ERROR] Local network error: {str(e)}")
            raise

        except ApiServerError as e:
            # Propagate API server errors
            logging.error(f"[ERROR] API server error: {str(e)}")
            raise

        except Exception as e:
            logging.error(f"[ERROR] API Exception: {str(e)}")
            raise Exception(str(e))

    def _parse_response(self, resp):
        """Parse response data - can be overridden by subclasses"""
        # The response is already the complete object, don't extract just the "data" field
        # as that would lose the outer structure (created timestamp, etc.)

        # Parse response using the provided model
        self.response = self.endpoint.response_model.model_validate(resp)
        logging.debug(f"[DEBUG] Parsed Response: {self.response}")
        return self.response


class TaskStatus(str, Enum):
    """Enum for task status values"""

    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"


class PollingOperation(Generic[T, R]):
    """
    Represents an asynchronous API operation that requires polling for completion.
    """

    def __init__(
        self,
        poll_endpoint: ApiEndpoint[EmptyRequest, R],
        completed_statuses: list,
        failed_statuses: list,
        status_extractor: Callable[[R], str],
        progress_extractor: Callable[[R], float] = None,
        result_url_extractor: Callable[[R], str] = None,
        request: Optional[T] = None,
        api_base: str | None = None,
        auth_token: Optional[str] = None,
        comfy_api_key: Optional[str] = None,
        auth_kwargs: Optional[Dict[str,str]] = None,
        poll_interval: float = 5.0,
        max_poll_attempts: int = 120,  # Default max polling attempts (10 minutes with 5s interval)
        max_retries: int = 3,  # Max retries per individual API call
        retry_delay: float = 1.0,
        retry_backoff_factor: float = 2.0,
        estimated_duration: Optional[float] = None,
        node_id: Optional[str] = None,
    ):
        self.poll_endpoint = poll_endpoint
        self.request = request
        self.api_base: str = api_base or args.comfy_api_base
        self.auth_token = auth_token
        self.comfy_api_key = comfy_api_key
        if auth_kwargs is not None:
            self.auth_token = auth_kwargs.get("auth_token", self.auth_token)
            self.comfy_api_key = auth_kwargs.get("comfy_api_key", self.comfy_api_key)
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff_factor = retry_backoff_factor
        self.estimated_duration = estimated_duration

        # Polling configuration
        self.status_extractor = status_extractor or (
            lambda x: getattr(x, "status", None)
        )
        self.progress_extractor = progress_extractor
        self.result_url_extractor = result_url_extractor
        self.node_id = node_id
        self.completed_statuses = completed_statuses
        self.failed_statuses = failed_statuses

        # For storing response data
        self.final_response = None
        self.error = None

    def execute(self, client: Optional[ApiClient] = None) -> R:
        """Execute the polling operation using the provided client. If failed, raise an exception."""
        try:
            if client is None:
                client = ApiClient(
                    base_url=self.api_base,
                    auth_token=self.auth_token,
                    comfy_api_key=self.comfy_api_key,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay,
                    retry_backoff_factor=self.retry_backoff_factor,
                )
            return self._poll_until_complete(client)
        except LocalNetworkError as e:
            # Provide clear message for local network issues
            raise Exception(
                f"Polling failed due to local network issues. Please check your internet connection. "
                f"Details: {str(e)}"
            ) from e
        except ApiServerError as e:
            # Provide clear message for API server issues
            raise Exception(
                f"Polling failed due to API server issues. The service may be experiencing problems. "
                f"Please try again later. Details: {str(e)}"
            ) from e
        except Exception as e:
            raise Exception(f"Error during polling: {str(e)}")

    def _display_text_on_node(self, text: str):
        """Sends text to the client which will be displayed on the node in the UI"""
        if not self.node_id:
            return

        PromptServer.instance.send_progress_text(text, self.node_id)

    def _display_time_progress_on_node(self, time_completed: int):
        if not self.node_id:
            return

        if self.estimated_duration is not None:
            estimated_time_remaining = max(
                0, int(self.estimated_duration) - int(time_completed)
            )
            message = f"Task in progress: {time_completed:.0f}s (~{estimated_time_remaining:.0f}s remaining)"
        else:
            message = f"Task in progress: {time_completed:.0f}s"
        self._display_text_on_node(message)

    def _check_task_status(self, response: R) -> TaskStatus:
        """Check task status using the status extractor function"""
        try:
            status = self.status_extractor(response)
            if status in self.completed_statuses:
                return TaskStatus.COMPLETED
            elif status in self.failed_statuses:
                return TaskStatus.FAILED
            return TaskStatus.PENDING
        except Exception as e:
            logging.error(f"Error extracting status: {e}")
            return TaskStatus.PENDING

    def _poll_until_complete(self, client: ApiClient) -> R:
        """Poll until the task is complete"""
        poll_count = 0
        consecutive_errors = 0
        max_consecutive_errors = min(5, self.max_retries * 2)  # Limit consecutive errors

        if self.progress_extractor:
            progress = utils.ProgressBar(PROGRESS_BAR_MAX)

        while poll_count < self.max_poll_attempts:
            try:
                poll_count += 1
                logging.debug(f"[DEBUG] Polling attempt #{poll_count}")

                request_dict = (
                    self.request.model_dump(exclude_none=True)
                    if self.request is not None
                    else None
                )

                if poll_count == 1:
                    logging.debug(
                        f"[DEBUG] Poll Request: {self.poll_endpoint.method.value} {self.poll_endpoint.path}"
                    )
                    logging.debug(
                        f"[DEBUG] Poll Request Data: {json.dumps(request_dict, indent=2) if request_dict else 'None'}"
                    )

                # Query task status
                resp = client.request(
                    method=self.poll_endpoint.method.value,
                    path=self.poll_endpoint.path,
                    params=self.poll_endpoint.query_params,
                    data=request_dict,
                )

                # Successfully got a response, reset consecutive error count
                consecutive_errors = 0

                # Parse response
                response_obj = self.poll_endpoint.response_model.model_validate(resp)

                # Check if task is complete
                status = self._check_task_status(response_obj)
                logging.debug(f"[DEBUG] Task Status: {status}")

                # If progress extractor is provided, extract progress
                if self.progress_extractor:
                    new_progress = self.progress_extractor(response_obj)
                    if new_progress is not None:
                        progress.update_absolute(new_progress, total=PROGRESS_BAR_MAX)

                if status == TaskStatus.COMPLETED:
                    message = "Task completed successfully"
                    if self.result_url_extractor:
                        result_url = self.result_url_extractor(response_obj)
                        if result_url:
                            message = f"Result URL: {result_url}"
                    else:
                        message = "Task completed successfully!"
                    logging.debug(f"[DEBUG] {message}")
                    self._display_text_on_node(message)
                    self.final_response = response_obj
                    if self.progress_extractor:
                        progress.update(100)
                    return self.final_response
                elif status == TaskStatus.FAILED:
                    message = f"Task failed: {json.dumps(resp)}"
                    logging.error(f"[DEBUG] {message}")
                    raise Exception(message)
                else:
                    logging.debug("[DEBUG] Task still pending, continuing to poll...")

                # Wait before polling again
                logging.debug(
                    f"[DEBUG] Waiting {self.poll_interval} seconds before next poll"
                )
                for i in range(int(self.poll_interval)):
                    time_completed = (poll_count * self.poll_interval) + i
                    self._display_time_progress_on_node(time_completed)
                    time.sleep(1)

            except (LocalNetworkError, ApiServerError) as e:
                # For network-related errors, increment error count and potentially abort
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise Exception(
                        f"Polling aborted after {consecutive_errors} consecutive network errors: {str(e)}"
                    ) from e

                # Log the error but continue polling
                logging.warning(
                    f"Network error during polling (attempt {poll_count}/{self.max_poll_attempts}): {str(e)}. "
                    f"Will retry in {self.poll_interval} seconds."
                )
                time.sleep(self.poll_interval)

            except Exception as e:
                # For other errors, increment count and potentially abort
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors or status == TaskStatus.FAILED:
                    raise Exception(
                        f"Polling aborted after {consecutive_errors} consecutive errors: {str(e)}"
                    ) from e

                logging.error(f"[DEBUG] Polling error: {str(e)}")
                logging.warning(
                    f"Error during polling (attempt {poll_count}/{self.max_poll_attempts}): {str(e)}. "
                    f"Will retry in {self.poll_interval} seconds."
                )
                time.sleep(self.poll_interval)

        # If we've exhausted all polling attempts
        raise Exception(
            f"Polling timed out after {poll_count} attempts ({poll_count * self.poll_interval} seconds). "
            f"The operation may still be running on the server but is taking longer than expected."
        )
