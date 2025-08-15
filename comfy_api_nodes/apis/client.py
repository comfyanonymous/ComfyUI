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
user_profile = await operation.execute(client=api_client)  # Returns immediately with the result


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
result = await operation.execute(client=api_client)  # Returns the final ImageGenerationResult when done
"""

from __future__ import annotations
import aiohttp
import asyncio
import logging
import io
import socket
from aiohttp.client_exceptions import ClientError, ClientResponseError
from typing import Dict, Type, Optional, Any, TypeVar, Generic, Callable, Tuple
from enum import Enum
import json
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
        session: Optional[aiohttp.ClientSession] = None,
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
        self._session: Optional[aiohttp.ClientSession] = session
        self._owns_session = session is None  # Track if we have to close it

    @staticmethod
    def _generate_operation_id(path: str) -> str:
        """Generates a unique operation ID for logging."""
        return f"{path.strip('/').replace('/', '_')}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _create_json_payload_args(
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        return {
            "json": data,
            "headers": headers,
        }

    def _create_form_data_args(
        self,
        data: Dict[str, Any] | None,
        files: Dict[str, Any] | None,
        headers: Optional[Dict[str, str]] = None,
        multipart_parser: Callable | None = None,
    ) -> Dict[str, Any]:
        if headers and "Content-Type" in headers:
            del headers["Content-Type"]

        if multipart_parser and data:
            data = multipart_parser(data)

        form = aiohttp.FormData(default_to_multipart=True)
        if data:  # regular text fields
            for k, v in data.items():
                if v is None:
                    continue  # aiohttp fails to serialize "None" values
                # aiohttp expects strings or bytes; convert enums etc.
                form.add_field(k, str(v) if not isinstance(v, (bytes, bytearray)) else v)

        if files:
            file_iter = files if isinstance(files, list) else files.items()
            for field_name, file_obj in file_iter:
                if file_obj is None:
                    continue  # aiohttp fails to serialize "None" values
                # file_obj can be (filename, bytes/io.BytesIO, content_type) tuple
                if isinstance(file_obj, tuple):
                    filename, file_value, content_type = self._unpack_tuple(file_obj)
                else:
                    file_value = file_obj
                    filename = getattr(file_obj, "name", field_name)
                    content_type = "application/octet-stream"

                form.add_field(
                    name=field_name,
                    value=file_value,
                    filename=filename,
                    content_type=content_type,
                )
        return {"data": form, "headers": headers or {}}

    @staticmethod
    def _create_urlencoded_form_data_args(
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

    async def _check_connectivity(self, target_url: str) -> Dict[str, bool]:
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
            "is_api_issue": False,
        }
        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get("https://www.google.com", ssl=self.verify_ssl) as resp:
                    results["internet_accessible"] = resp.status < 500
            except (ClientError, asyncio.TimeoutError, socket.gaierror):
                results["is_local_issue"] = True
                return results  # cannot reach the internet – early exit

            # Now check API health endpoint
            parsed = urlparse(target_url)
            health_url = f"{parsed.scheme}://{parsed.netloc}/health"
            try:
                async with session.get(health_url, ssl=self.verify_ssl) as resp:
                    results["api_accessible"] = resp.status < 500
            except ClientError:
                pass  # leave as False

        results["is_api_issue"] = results["internet_accessible"] and not results["api_accessible"]
        return results

    async def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any] | list[tuple[str, Any]]] = None,
        headers: Optional[Dict[str, str]] = None,
        content_type: str = "application/json",
        multipart_parser: Callable | None = None,
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

        # Build full URL and merge headers
        relative_path = path.lstrip("/")
        url = urljoin(self.base_url, relative_path)
        self._check_auth(self.auth_token, self.comfy_api_key)

        request_headers = self.get_headers()
        if headers:
            request_headers.update(headers)
        if files:
            request_headers.pop("Content-Type", None)
        if params:
            params = {k: v for k, v in params.items() if v is not None}  # aiohttp fails to serialize None values

        logging.debug(f"[DEBUG] Request Headers: {request_headers}")
        logging.debug(f"[DEBUG] Files: {files}")
        logging.debug(f"[DEBUG] Params: {params}")
        logging.debug(f"[DEBUG] Data: {data}")

        if content_type == "application/x-www-form-urlencoded":
            payload_args = self._create_urlencoded_form_data_args(data or {}, request_headers)
        elif content_type == "multipart/form-data":
            payload_args = self._create_form_data_args(data, files, request_headers, multipart_parser)
        else:
            payload_args = self._create_json_payload_args(data, request_headers)

        operation_id = self._generate_operation_id(path)
        request_logger.log_request_response(
            operation_id=operation_id,
            request_method=method,
            request_url=url,
            request_headers=request_headers,
            request_params=params,
            request_data=data if content_type == "application/json" else "[form-data or other]",
        )

        session = await self._get_session()
        try:
            async with session.request(
                method,
                url,
                params=params,
                ssl=self.verify_ssl,
                **payload_args,
            ) as resp:
                if resp.status >= 400:
                    try:
                        error_data = await resp.json()
                    except (aiohttp.ContentTypeError, json.JSONDecodeError):
                        error_data = await resp.text()

                    return await self._handle_http_error(
                        ClientResponseError(resp.request_info, resp.history, status=resp.status, message=error_data),
                        operation_id,
                        method,
                        url,
                        params,
                        data,
                        files,
                        headers,
                        content_type,
                        multipart_parser,
                        retry_count=retry_count,
                        response_content=error_data,
                    )

                # Success – parse JSON (safely) and log
                try:
                    payload = await resp.json()
                    response_content_to_log = payload
                except (aiohttp.ContentTypeError, json.JSONDecodeError):
                    payload = {}
                    response_content_to_log = await resp.text()

                request_logger.log_request_response(
                    operation_id=operation_id,
                    request_method=method,
                    request_url=url,
                    response_status_code=resp.status,
                    response_headers=dict(resp.headers),
                    response_content=response_content_to_log,
                )
                return payload

        except (ClientError, asyncio.TimeoutError, socket.gaierror) as e:
            # Treat as *connection* problem – optionally retry, else escalate
            if retry_count < self.max_retries:
                delay = self.retry_delay * (self.retry_backoff_factor ** retry_count)
                logging.warning("Connection error. Retrying in %.2fs (%s/%s): %s", delay, retry_count + 1,
                                self.max_retries, str(e))
                await asyncio.sleep(delay)
                return await self.request(
                    method,
                    path,
                    params=params,
                    data=data,
                    files=files,
                    headers=headers,
                    content_type=content_type,
                    multipart_parser=multipart_parser,
                    retry_count=retry_count + 1,
                )
            # One final connectivity check for diagnostics
            connectivity = await self._check_connectivity(self.base_url)
            if connectivity["is_local_issue"]:
                raise LocalNetworkError(
                    "Unable to connect to the API server due to local network issues. "
                    "Please check your internet connection and try again."
                ) from e
            raise ApiServerError(
                f"The API server at {self.base_url} is currently unreachable. "
                f"The service may be experiencing issues. Please try again later."
            ) from e

    @staticmethod
    def _check_auth(auth_token, comfy_api_key):
        """Verify that an auth token is present or comfy_api_key is present"""
        if auth_token is None and comfy_api_key is None:
            raise Exception("Unauthorized: Please login first to use this node.")
        return auth_token or comfy_api_key

    @staticmethod
    async def upload_file(
        upload_url: str,
        file: io.BytesIO | str,
        content_type: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff_factor: float = 2.0,
    ) -> aiohttp.ClientResponse:
        """Upload a file to the API with retry logic.

        Args:
            upload_url: The URL to upload to
            file: Either a file path string, BytesIO object, or tuple of (file_path, filename)
            content_type: Optional mime type to set for the upload
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            retry_backoff_factor: Multiplier for the delay after each retry
        """
        headers: Dict[str, str] = {}
        skip_auto_headers: set[str] = set()
        if content_type:
            headers["Content-Type"] = content_type
        else:
            # tell aiohttp not to add Content-Type that will break the request signature and result in a 403 status.
            skip_auto_headers.add("Content-Type")

        # Extract file bytes
        if isinstance(file, io.BytesIO):
            file.seek(0)
            data = file.read()
        elif isinstance(file, str):
            with open(file, "rb") as f:
                data = f.read()
        else:
            raise ValueError("File must be BytesIO or str path")

        operation_id = f"upload_{upload_url.split('/')[-1]}_{uuid.uuid4().hex[:8]}"
        request_logger.log_request_response(
            operation_id=operation_id,
            request_method="PUT",
            request_url=upload_url,
            request_headers=headers,
            request_data=f"[File data {len(data)} bytes]",
        )

        delay = retry_delay
        for attempt in range(max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=None)  # honour server side timeouts
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.put(
                        upload_url, data=data, headers=headers, skip_auto_headers=skip_auto_headers,
                    ) as resp:
                        resp.raise_for_status()
                        request_logger.log_request_response(
                            operation_id=operation_id,
                            request_method="PUT",
                            request_url=upload_url,
                            response_status_code=resp.status,
                            response_headers=dict(resp.headers),
                            response_content="File uploaded successfully.",
                        )
                        return resp
            except (ClientError, asyncio.TimeoutError) as e:
                request_logger.log_request_response(
                    operation_id=operation_id,
                    request_method="PUT",
                    request_url=upload_url,
                    response_status_code=e.status if hasattr(e, "status") else None,
                    response_headers=dict(e.headers) if getattr(e, "headers") else None,
                    response_content=None,
                    error_message=f"{type(e).__name__}: {str(e)}",
                )
                if attempt < max_retries:
                    logging.warning(
                        "Upload failed (%s/%s). Retrying in %.2fs. %s", attempt + 1, max_retries, delay, str(e)
                    )
                    await asyncio.sleep(delay)
                    delay *= retry_backoff_factor
                else:
                    raise NetworkError(f"Failed to upload file after {max_retries + 1} attempts: {e}") from e

    async def _handle_http_error(
        self,
        exc: ClientResponseError,
        operation_id: str,
        *req_meta,
        retry_count: int,
        response_content: dict | str = "",
    ) -> Dict[str, Any]:
        status_code = exc.status
        if status_code == 401:
            user_friendly = "Unauthorized: Please login first to use this node."
        elif status_code == 402:
            user_friendly = "Payment Required: Please add credits to your account to use this node."
        elif status_code == 409:
            user_friendly = "There is a problem with your account. Please contact support@comfy.org."
        elif status_code == 429:
            user_friendly = "Rate Limit Exceeded: Please try again later."
        else:
            if isinstance(response_content, dict):
                if "error" in response_content and "message" in response_content["error"]:
                    user_friendly = f"API Error: {response_content['error']['message']}"
                    if "type" in response_content["error"]:
                        user_friendly += f" (Type: {response_content['error']['type']})"
                else: # Handle cases where error is just a JSON dict with unknown format
                    user_friendly = f"API Error: {json.dumps(response_content)}"
            else:
                if len(response_content) < 200:  # Arbitrary limit for display
                    user_friendly = f"API Error (raw): {response_content}"
                else:
                    user_friendly = f"API Error (raw, status {response_content})"

        request_logger.log_request_response(
            operation_id=operation_id,
            request_method=req_meta[0],
            request_url=req_meta[1],
            response_status_code=exc.status,
            response_headers=dict(req_meta[5]) if req_meta[5] else None,
            response_content=response_content,
            error_message=f"HTTP Error {exc.status}",
        )

        logging.debug(f"[DEBUG] API Error: {user_friendly} (Status: {status_code})")
        if response_content:
            logging.debug(f"[DEBUG] Response content: {response_content}")

        # Retry if eligible
        if status_code in self.retry_status_codes and retry_count < self.max_retries:
            delay = self.retry_delay * (self.retry_backoff_factor ** retry_count)
            logging.warning(
                "HTTP error %s. Retrying in %.2fs (%s/%s)",
                status_code,
                delay,
                retry_count + 1,
                self.max_retries,
            )
            await asyncio.sleep(delay)
            return await self.request(
                req_meta[0],  # method
                req_meta[1].replace(self.base_url, ""),  # path
                params=req_meta[2],
                data=req_meta[3],
                files=req_meta[4],
                headers=req_meta[5],
                content_type=req_meta[6],
                multipart_parser=req_meta[7],
                retry_count=retry_count + 1,
            )

        raise Exception(user_friendly) from exc

    @staticmethod
    def _unpack_tuple(t):
        """Helper to normalise (filename, file, content_type) tuples."""
        if len(t) == 3:
            return t
        elif len(t) == 2:
            return t[0], t[1], "application/octet-stream"
        else:
            raise ValueError("files tuple must be (filename, file[, content_type])")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "ApiClient":
        """Allow usage as async‑context‑manager – ensures clean teardown"""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()


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
    """Represents a single synchronous API operation."""

    def __init__(
        self,
        endpoint: ApiEndpoint[T, R],
        request: T,
        files: Optional[Dict[str, Any] | list[tuple[str, Any]]] = None,
        api_base: str | None = None,
        auth_token: Optional[str] = None,
        comfy_api_key: Optional[str] = None,
        auth_kwargs: Optional[Dict[str, str]] = None,
        timeout: float = 604800.0,
        verify_ssl: bool = True,
        content_type: str = "application/json",
        multipart_parser: Callable | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff_factor: float = 2.0,
    ) -> None:
        self.endpoint = endpoint
        self.request = request
        self.files = files
        self.api_base: str = api_base or args.comfy_api_base
        self.auth_token = auth_token
        self.comfy_api_key = comfy_api_key
        if auth_kwargs is not None:
            self.auth_token = auth_kwargs.get("auth_token", self.auth_token)
            self.comfy_api_key = auth_kwargs.get("comfy_api_key", self.comfy_api_key)
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.content_type = content_type
        self.multipart_parser = multipart_parser
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff_factor = retry_backoff_factor

    async def execute(self, client: Optional[ApiClient] = None) -> R:
        owns_client = client is None
        if owns_client:
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

        try:
            request_dict: Optional[Dict[str, Any]]
            if isinstance(self.request, EmptyRequest):
                request_dict = None
            else:
                request_dict = self.request.model_dump(exclude_none=True)
                for k, v in list(request_dict.items()):
                    if isinstance(v, Enum):
                        request_dict[k] = v.value

            logging.debug(
                f"[DEBUG] API Request: {self.endpoint.method.value} {self.endpoint.path}"
            )
            logging.debug(f"[DEBUG] Request Data: {json.dumps(request_dict, indent=2)}")
            logging.debug(f"[DEBUG] Query Params: {self.endpoint.query_params}")

            response_json = await client.request(
                self.endpoint.method.value,
                self.endpoint.path,
                params=self.endpoint.query_params,
                data=request_dict,
                files=self.files,
                content_type=self.content_type,
                multipart_parser=self.multipart_parser,
            )

            logging.debug("=" * 50)
            logging.debug("[DEBUG] RESPONSE DETAILS:")
            logging.debug("[DEBUG] Status Code: 200 (Success)")
            logging.debug(f"[DEBUG] Response Body: {json.dumps(response_json, indent=2)}")
            logging.debug("=" * 50)

            parsed_response = self.endpoint.response_model.model_validate(response_json)
            logging.debug(f"[DEBUG] Parsed Response: {parsed_response}")
            return parsed_response
        finally:
            if owns_client:
                await client.close()


class TaskStatus(str, Enum):
    """Enum for task status values"""

    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"


class PollingOperation(Generic[T, R]):
    """Represents an asynchronous API operation that requires polling for completion."""

    def __init__(
        self,
        poll_endpoint: ApiEndpoint[EmptyRequest, R],
        completed_statuses: list[str],
        failed_statuses: list[str],
        status_extractor: Callable[[R], str],
        progress_extractor: Callable[[R], float] | None = None,
        result_url_extractor: Callable[[R], str] | None = None,
        request: Optional[T] = None,
        api_base: str | None = None,
        auth_token: Optional[str] = None,
        comfy_api_key: Optional[str] = None,
        auth_kwargs: Optional[Dict[str, str]] = None,
        poll_interval: float = 5.0,
        max_poll_attempts: int = 120,  # Default max polling attempts (10 minutes with 5s interval)
        max_retries: int = 3,  # Max retries per individual API call
        retry_delay: float = 1.0,
        retry_backoff_factor: float = 2.0,
        estimated_duration: Optional[float] = None,
        node_id: Optional[str] = None,
    ) -> None:
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
        self.status_extractor = status_extractor or (lambda x: getattr(x, "status", None))
        self.progress_extractor = progress_extractor
        self.result_url_extractor = result_url_extractor
        self.node_id = node_id
        self.completed_statuses = completed_statuses
        self.failed_statuses = failed_statuses
        self.final_response: Optional[R] = None

    async def execute(self, client: Optional[ApiClient] = None) -> R:
        owns_client = client is None
        if owns_client:
            client = ApiClient(
                base_url=self.api_base,
                auth_token=self.auth_token,
                comfy_api_key=self.comfy_api_key,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                retry_backoff_factor=self.retry_backoff_factor,
            )
        try:
            return await self._poll_until_complete(client)
        finally:
            if owns_client:
                await client.close()

    def _display_text_on_node(self, text: str):
        if not self.node_id:
            return
        PromptServer.instance.send_progress_text(text, self.node_id)

    def _display_time_progress_on_node(self, time_completed: int | float):
        if not self.node_id:
            return
        if self.estimated_duration is not None:
            remaining = max(0, int(self.estimated_duration) - time_completed)
            message = f"Task in progress: {time_completed}s (~{remaining}s remaining)"
        else:
            message = f"Task in progress: {time_completed}s"
        self._display_text_on_node(message)

    def _check_task_status(self, response: R) -> TaskStatus:
        try:
            status = self.status_extractor(response)
            if status in self.completed_statuses:
                return TaskStatus.COMPLETED
            if status in self.failed_statuses:
                return TaskStatus.FAILED
            return TaskStatus.PENDING
        except Exception as e:
            logging.error("Error extracting status: %s", e)
            return TaskStatus.PENDING

    async def _poll_until_complete(self, client: ApiClient) -> R:
        """Poll until the task is complete"""
        consecutive_errors = 0
        max_consecutive_errors = min(5, self.max_retries * 2)  # Limit consecutive errors

        if self.progress_extractor:
            progress = utils.ProgressBar(PROGRESS_BAR_MAX)

        status = TaskStatus.PENDING
        for poll_count in range(1, self.max_poll_attempts + 1):
            try:
                logging.debug(f"[DEBUG] Polling attempt #{poll_count}")

                request_dict = (
                    None if self.request is None else self.request.model_dump(exclude_none=True)
                )

                if poll_count == 1:
                    logging.debug(
                        f"[DEBUG] Poll Request: {self.poll_endpoint.method.value} {self.poll_endpoint.path}"
                    )
                    logging.debug(
                        f"[DEBUG] Poll Request Data: {json.dumps(request_dict, indent=2) if request_dict else 'None'}"
                    )

                # Query task status
                resp = await client.request(
                    self.poll_endpoint.method.value,
                    self.poll_endpoint.path,
                    params=self.poll_endpoint.query_params,
                    data=request_dict,
                )
                consecutive_errors = 0  # reset on success
                response_obj: R = self.poll_endpoint.response_model.model_validate(resp)

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
                    logging.debug(f"[DEBUG] {message}")
                    self._display_text_on_node(message)
                    self.final_response = response_obj
                    if self.progress_extractor:
                        progress.update(100)
                    return self.final_response
                if status == TaskStatus.FAILED:
                    message = f"Task failed: {json.dumps(resp)}"
                    logging.error(f"[DEBUG] {message}")
                    raise Exception(message)
                logging.debug("[DEBUG] Task still pending, continuing to poll...")
                # Task pending – wait
                for i in range(int(self.poll_interval)):
                    self._display_time_progress_on_node((poll_count - 1) * self.poll_interval + i)
                    await asyncio.sleep(1)

            except (LocalNetworkError, ApiServerError, NetworkError) as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise Exception(
                        f"Polling aborted after {consecutive_errors} network errors: {str(e)}"
                    ) from e
                logging.warning("Network error (%s/%s): %s", consecutive_errors, max_consecutive_errors, str(e))
                await asyncio.sleep(self.poll_interval)
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
                await asyncio.sleep(self.poll_interval)

        # If we've exhausted all polling attempts
        raise Exception(
            f"Polling timed out after {self.max_poll_attempts} attempts (" f"{self.max_poll_attempts * self.poll_interval} seconds). "
            "The operation may still be running on the server but is taking longer than expected."
        )
