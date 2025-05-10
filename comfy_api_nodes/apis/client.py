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
from typing import Dict, Type, Optional, Any, TypeVar, Generic, Callable
from enum import Enum
import json
import requests
from urllib.parse import urljoin
from pydantic import BaseModel, Field

from comfy.cli_args import args
from comfy import utils

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R", bound=BaseModel)
P = TypeVar("P", bound=BaseModel)  # For poll response

PROGRESS_BAR_MAX = 100


class EmptyRequest(BaseModel):
    """Base class for empty request bodies.
    For GET requests, fields will be sent as query parameters."""

    pass


class UploadRequest(BaseModel):
    file_name: str = Field(..., description="Filename to upload")
    content_type: str | None = Field(
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
    Client for making HTTP requests to an API with authentication and error handling.
    """

    def __init__(
        self,
        base_url: str,
        auth_token: Optional[str] = None,
        comfy_api_key: Optional[str] = None,
        timeout: float = 3600.0,
        verify_ssl: bool = True,
    ):
        self.base_url = base_url
        self.auth_token = auth_token
        self.comfy_api_key = comfy_api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl

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
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path (will be joined with base_url)
            params: Query parameters
            data: body data
            files: Files to upload
            headers: Additional headers
            content_type: Content type of the request. Defaults to application/json.

        Returns:
            Parsed JSON response

        Raises:
            requests.RequestException: If the request fails
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

        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                timeout=self.timeout,
                verify=self.verify_ssl,
                **payload_args,
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
                logging.debug(
                    f"[DEBUG] Failed to parse error response: {str(json_error)}"
                )

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
    ):
        """Upload a file to the API. Make sure the file has a filename equal to what the url expects.

        Args:
            upload_url: The URL to upload to
            file: Either a file path string, BytesIO object, or tuple of (file_path, filename)
            mime_type: Optional mime type to set for the upload
        """
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type

        if isinstance(file, io.BytesIO):
            file.seek(0)  # Ensure we're at the start of the file
            data = file.read()
            return requests.put(upload_url, data=data, headers=headers)
        elif isinstance(file, str):
            with open(file, "rb") as f:
                data = f.read()
                return requests.put(upload_url, data=data, headers=headers)


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
    def execute(self, client: Optional[ApiClient] = None) -> R:
        """Execute the API operation using the provided client or create one"""
        try:
            # Create client if not provided
            if client is None:
                client = ApiClient(
                    base_url=self.api_base,
                    auth_token=self.auth_token,
                    comfy_api_key=self.comfy_api_key,
                    timeout=self.timeout,
                    verify_ssl=self.verify_ssl,
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

            # Make the request
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

        except Exception as e:
            logging.error(f"[DEBUG] API Exception: {str(e)}")
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
        request: Optional[T] = None,
        api_base: str | None = None,
        auth_token: Optional[str] = None,
        comfy_api_key: Optional[str] = None,
        auth_kwargs: Optional[Dict[str,str]] = None,
        poll_interval: float = 5.0,
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

        # Polling configuration
        self.status_extractor = status_extractor or (
            lambda x: getattr(x, "status", None)
        )
        self.progress_extractor = progress_extractor
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
                )
            return self._poll_until_complete(client)
        except Exception as e:
            raise Exception(f"Error during polling: {str(e)}")

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
        if self.progress_extractor:
            progress = utils.ProgressBar(PROGRESS_BAR_MAX)

        while True:
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
                    logging.debug("[DEBUG] Task completed successfully")
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
                time.sleep(self.poll_interval)

            except Exception as e:
                logging.error(f"[DEBUG] Polling error: {str(e)}")
                raise Exception(f"Error while polling: {str(e)}")
