"""
Dynamic partner node loading from comfy-api.

This module fetches object_info from comfy-api's /object_info endpoint and
creates proxy node classes that can be used in ComfyUI. These nodes are
dynamically generated based on the node definitions returned by the API.

The execution is handled by a generic executor that:
1. Serializes inputs according to proxy config (e.g., images to base64)
2. POSTs to the dynamic proxy endpoint
3. Receives bytes back and converts to appropriate output type
"""

import asyncio
import logging
from io import BytesIO
from typing import Any, Optional

import aiohttp

from comfy_api.latest import IO, ComfyExtension, Input, InputImpl
from comfy_api_nodes.util import (
    ApiEndpoint,
    default_base_url,
    sync_op_raw,
    tensor_to_base64_string,
)

# Cache for fetched object_info
_object_info_cache: dict[str, Any] | None = None


async def fetch_dynamic_object_info() -> dict[str, Any]:
    """Fetch object_info from comfy-api's /object_info endpoint."""
    global _object_info_cache
    
    if _object_info_cache is not None:
        return _object_info_cache
    
    url = f"{default_base_url()}/object_info"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    _object_info_cache = await resp.json()
                    logging.info(f"Fetched {len(_object_info_cache)} dynamic partner nodes from {url}")
                    return _object_info_cache
                else:
                    logging.warning(f"Failed to fetch dynamic object_info: HTTP {resp.status}")
                    return {}
    except Exception as e:
        logging.warning(f"Failed to fetch dynamic object_info from {url}: {e}")
        return {}


def create_dynamic_node_class(node_id: str, node_info: dict[str, Any]) -> type[IO.ComfyNode]:
    """
    Create a dynamic ComfyNode class from an object_info definition.
    
    The created node will proxy execution to comfy-api which handles the
    actual API calls to partner services.
    """
    
    # Extract node metadata
    display_name = node_info.get("display_name", node_id)
    category = node_info.get("category", "api node/dynamic")
    description = node_info.get("description", "")
    is_api_node = node_info.get("api_node", True)
    
    # Get proxy configuration
    proxy_config = node_info.get("proxy", {})
    
    # Parse inputs from object_info format
    input_def = node_info.get("input", {})
    input_order = node_info.get("input_order", {})
    
    # Parse outputs
    output_types = node_info.get("output", [])
    output_names = node_info.get("output_name", output_types)
    output_is_list = node_info.get("output_is_list", [False] * len(output_types))
    
    # Build inputs list for the schema
    inputs = []
    
    # Process required inputs
    required_order = input_order.get("required", [])
    required_inputs = input_def.get("required", {})
    for input_name in required_order:
        if input_name in required_inputs:
            input_spec = required_inputs[input_name]
            inp = _parse_input_spec(input_name, input_spec, optional=False)
            if inp:
                inputs.append(inp)
    
    # Process optional inputs
    optional_order = input_order.get("optional", [])
    optional_inputs = input_def.get("optional", {})
    for input_name in optional_order:
        if input_name in optional_inputs:
            input_spec = optional_inputs[input_name]
            inp = _parse_input_spec(input_name, input_spec, optional=True)
            if inp:
                inputs.append(inp)
    
    # Build outputs list
    outputs = []
    for i, output_type in enumerate(output_types):
        name = output_names[i] if i < len(output_names) else output_type
        is_list = output_is_list[i] if i < len(output_is_list) else False
        out = _parse_output_spec(name, output_type, is_list)
        if out:
            outputs.append(out)
    
    # Create the dynamic node class
    class DynamicProxyNode(IO.ComfyNode):
        @classmethod
        def define_schema(cls):
            return IO.Schema(
                node_id=node_id,
                display_name=display_name,
                category=category,
                description=description,
                inputs=inputs,
                outputs=outputs,
                hidden=[
                    IO.Hidden.auth_token_comfy_org,
                    IO.Hidden.api_key_comfy_org,
                    IO.Hidden.unique_id,
                ],
                is_api_node=is_api_node,
            )
        
        @classmethod
        async def execute(cls, **kwargs):
            """Execute the node by proxying to comfy-api's dynamic endpoint."""
            return await _execute_dynamic_node(cls, node_id, proxy_config, kwargs)
    
    # Set module info for the dynamic class
    DynamicProxyNode.__name__ = node_id
    DynamicProxyNode.__qualname__ = node_id
    DynamicProxyNode.RELATIVE_PYTHON_MODULE = "comfy_api_nodes.dynamic_nodes"
    
    return DynamicProxyNode


def _parse_input_spec(name: str, spec: list, optional: bool) -> Optional[Input]:
    """Parse an input specification from object_info format."""
    if not spec or len(spec) < 1:
        return None
    
    io_type = spec[0]
    options = spec[1] if len(spec) > 1 else {}
    
    # Handle combo inputs (list of options)
    if isinstance(io_type, list):
        return IO.Combo.Input(
            name,
            options=io_type,
            default=options.get("default"),
            tooltip=options.get("tooltip"),
            optional=optional,
        )
    
    # Handle standard types
    if io_type == "STRING":
        return IO.String.Input(
            name,
            default=options.get("default", ""),
            multiline=options.get("multiline", False),
            tooltip=options.get("tooltip"),
            optional=optional,
        )
    elif io_type == "INT":
        return IO.Int.Input(
            name,
            default=options.get("default", 0),
            min=options.get("min", 0),
            max=options.get("max", 2147483647),
            step=options.get("step", 1),
            tooltip=options.get("tooltip"),
            optional=optional,
        )
    elif io_type == "FLOAT":
        return IO.Float.Input(
            name,
            default=options.get("default", 0.0),
            min=options.get("min", 0.0),
            max=options.get("max", 1.0),
            step=options.get("step", 0.01),
            tooltip=options.get("tooltip"),
            optional=optional,
        )
    elif io_type == "BOOLEAN":
        return IO.Boolean.Input(
            name,
            default=options.get("default", False),
            tooltip=options.get("tooltip"),
            optional=optional,
        )
    elif io_type == "IMAGE":
        return IO.Image.Input(
            name,
            tooltip=options.get("tooltip"),
            optional=optional,
        )
    elif io_type == "VIDEO":
        return IO.Video.Input(
            name,
            tooltip=options.get("tooltip"),
            optional=optional,
        )
    elif io_type == "AUDIO":
        return IO.Audio.Input(
            name,
            tooltip=options.get("tooltip"),
            optional=optional,
        )
    else:
        # Generic/custom type - treat as Any
        logging.debug(f"Unknown input type {io_type} for {name}, skipping")
        return None


def _parse_output_spec(name: str, output_type: str, is_list: bool) -> Optional[IO.Output]:
    """Parse an output specification."""
    if output_type == "VIDEO":
        return IO.Video.Output(display_name=name)
    elif output_type == "IMAGE":
        return IO.Image.Output(display_name=name)
    elif output_type == "AUDIO":
        return IO.Audio.Output(display_name=name)
    elif output_type == "STRING":
        return IO.String.Output(display_name=name)
    elif output_type == "INT":
        return IO.Int.Output(display_name=name)
    elif output_type == "FLOAT":
        return IO.Float.Output(display_name=name)
    else:
        logging.debug(f"Unknown output type {output_type} for {name}, using generic")
        return IO.Video.Output(display_name=name)


async def _execute_dynamic_node(
    cls: type[IO.ComfyNode],
    node_id: str,
    proxy_config: dict[str, Any],
    inputs: dict[str, Any],
) -> IO.NodeOutput:
    """
    Execute a dynamic node using the generic proxy endpoint.
    
    This is the core executor that:
    1. Serializes inputs according to proxy config
    2. POSTs to the dynamic proxy endpoint  
    3. Receives bytes back and converts to output type
    """
    
    # Get proxy endpoint from config, or use default pattern
    endpoint = proxy_config.get("endpoint", f"/proxy/dynamic/{node_id}")
    input_serialization = proxy_config.get("input_serialization", {})
    output_type = proxy_config.get("output_type", "video")
    
    # Serialize inputs
    serialized_inputs = _serialize_inputs(inputs, input_serialization)
    
    # Build request with inputs wrapper
    request_data = {"inputs": serialized_inputs}
    
    # Call the dynamic proxy endpoint
    response = await sync_op_raw(
        cls,
        ApiEndpoint(endpoint, "POST"),
        data=request_data,
        as_binary=True,
        max_retries=1,
    )
    
    # Convert response to appropriate output type
    return _deserialize_output(response, output_type)


def _serialize_inputs(inputs: dict[str, Any], serialization_config: dict[str, str]) -> dict[str, Any]:
    """
    Serialize inputs according to the configuration.
    
    For example, IMAGE inputs may need to be converted to base64.
    """
    result = {}
    
    for key, value in inputs.items():
        if value is None:
            continue
            
        serialization_type = serialization_config.get(key)
        
        if serialization_type == "base64":
            # Convert tensor to base64 string
            if hasattr(value, 'shape'):  # It's a tensor
                result[key] = tensor_to_base64_string(value)
            else:
                result[key] = value
        else:
            # Pass through as-is
            result[key] = value
    
    return result


def _deserialize_output(data: bytes, output_type: str) -> IO.NodeOutput:
    """
    Convert response bytes to the appropriate output type.
    """
    if output_type == "video":
        return IO.NodeOutput(InputImpl.VideoFromFile(BytesIO(data)))
    elif output_type == "image":
        return IO.NodeOutput(InputImpl.ImageFromFile(BytesIO(data)))
    elif output_type == "audio":
        return IO.NodeOutput(InputImpl.AudioFromFile(BytesIO(data)))
    else:
        # Default to video
        return IO.NodeOutput(InputImpl.VideoFromFile(BytesIO(data)))


# ============================================================================
# Extension Registration
# ============================================================================

class DynamicPartnerNodesExtension(ComfyExtension):
    """Extension that dynamically loads partner nodes from comfy-api."""
    
    @classmethod
    def on_register(cls):
        """Called when the extension is registered. Fetch and create nodes."""
        try:
            # Fetch object_info synchronously at startup
            loop = asyncio.new_event_loop()
            try:
                object_info = loop.run_until_complete(fetch_dynamic_object_info())
            finally:
                loop.close()
            
            if not object_info:
                logging.warning("No dynamic partner nodes loaded from comfy-api")
                return []
            
            # Create node classes for each definition
            node_classes = []
            for node_id, node_info in object_info.items():
                # Only create nodes that have proxy config (dynamic nodes)
                if "proxy" in node_info:
                    try:
                        node_class = create_dynamic_node_class(node_id, node_info)
                        node_classes.append(node_class)
                        logging.info(f"Created dynamic node: {node_id}")
                    except Exception as e:
                        logging.warning(f"Failed to create dynamic node {node_id}: {e}")
            
            return node_classes
            
        except Exception as e:
            logging.error(f"Failed to load dynamic partner nodes: {e}")
            return []


# Register the extension
NODES = DynamicPartnerNodesExtension
