# generate_text.py
import os
import sys
import json
import base64
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, AsyncGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import ComfyUI-specific modules
try:
    import folder_paths
    from server import PromptServer # <-- Import PromptServer
except ImportError:
    logger.warning("Could not import folder_paths or server. Make sure ComfyUI environment is set up.")
    folder_paths = None
    PromptServer = None

# Check for required dependencies
missing_deps = []
try:
    import aiohttp
except ImportError:
    missing_deps.append("aiohttp")

if missing_deps:
    logger.warning(f"Missing dependencies: {', '.join(missing_deps)}. Some functionality may not work.")
    logger.warning("Please install missing dependencies: pip install " + " ".join(missing_deps))

# Import utility functions (assuming they exist)
from send_request import send_request, run_async, is_gpt5_model # Keep non-streaming version if needed
from api.openai_api import send_openai_responses_stream
from api.gemini_api import send_gemini_request_stream
from llmtoolkit_utils import query_local_ollama_models, ensure_ollama_server, ensure_ollama_model, get_api_key

# Local transformers streaming (optional) - removed unused import
# The send_transformers_request_stream function was imported but never used
send_transformers_request_stream = None  # type: ignore

# Payload helper to embed context into a string subclass
from context_payload import ContextPayload

# -----------------------------------------------------------------------------
# Helpers: Lazy OpenCV import and video -> base64 frame extraction
# -----------------------------------------------------------------------------
_cv2_ref = None

def _get_cv2():
    global _cv2_ref
    if _cv2_ref is None:
        try:
            import cv2 as _cv2  # type: ignore
            _cv2_ref = _cv2
        except Exception:
            _cv2_ref = None
            logger.warning("cv2 not available. Video file frame extraction disabled.")
    return _cv2_ref


def _is_video_file(path: str) -> bool:
    try:
        ext = os.path.splitext(path)[1].lower()
    except Exception:
        return False
    return ext in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _extract_video_file_frames_as_b64(
    video_path: str,
    max_frames: int = 5,
    stride: int = 16,
) -> List[str]:
    """Extract up to `max_frames` JPEG base64 frames from a local video file.

    - Uses a simple stride to subsample frames.
    - Falls back to the first N frames if the video is very short.
    - Returns an empty list on any error or if cv2 is unavailable.
    """
    cv2 = _get_cv2()
    if cv2 is None:
        return []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Could not open video file: %s", video_path)
            return []

        frames: List[str] = []
        frame_index = 0
        picked = 0

        # Try stride sampling first
        while picked < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                break
            success, buf = cv2.imencode(".jpg", frame)
            if success:
                frames.append(base64.b64encode(buf.tobytes()).decode("ascii"))
                picked += 1
            frame_index += stride

        # If we got nothing with stride, try the first sequential frames
        if not frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            count = 0
            while count < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                success, buf = cv2.imencode(".jpg", frame)
                if success:
                    frames.append(base64.b64encode(buf.tobytes()).decode("ascii"))
                    count += 1

        cap.release()
        if frames:
            logger.debug("Extracted %d frame(s) from video file %s", len(frames), video_path)
        return frames
    except Exception as e:
        logger.warning("Error extracting frames from %s: %s", video_path, e, exc_info=True)
        return []

# --- NEW STREAMING REQUEST FUNCTION (Example for Ollama) ---
# IMPORTANT: This needs to be adapted based on the actual API structure of the provider!
async def send_request_stream(
    llm_provider: str,
    base_ip: str,
    port: str,
    llm_model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, str]],
    seed: Optional[int] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    random: bool = False,
    top_k: int = 40,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    stop: Optional[List[str]] = None,
    keep_alive: Union[bool, str] = True,
    llm_api_key: Optional[str] = None,
    timeout: int = 120, # Add a timeout for the connection
    base64_images: Optional[List[str]] = None,
) -> AsyncGenerator[str, None]:
    """
    Sends a streaming request to an LLM provider (Example for Ollama).
    Yields text chunks as they are received.
    """
    provider_lower = llm_provider.lower()

    if provider_lower in ["openai", "openrouter"]:
        # --- OpenAI & OpenRouter Specific Streaming Logic ---
        if not llm_api_key:
            logger.error(f"{llm_provider} streaming requested but no API key supplied.")
            yield f"[{llm_provider} Error: API key missing]"
            return

        # Check if this is a GPT-5 model and use Responses API (OpenAI only)
        if provider_lower == "openai" and is_gpt5_model(llm_model):
            logger.info(f"Detected GPT-5 model: {llm_model}, using Responses API for streaming")
            try:
                async for chunk in send_openai_responses_stream(
                    api_url="https://api.openai.com/v1/responses",
                    base64_images=base64_images,
                    model=llm_model,
                    system_message=system_message,
                    user_message=user_message,
                    messages=messages,
                    api_key=llm_api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                ):
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"GPT-5 Responses stream failed: {e}, falling back to Chat Completions")
                # Fall through to regular OpenAI streaming below

        api_url = "https://openrouter.ai/api/v1/chat/completions" if provider_lower == "openrouter" else "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {llm_api_key}",
            "Content-Type": "application/json",
        }
        # Build message list if not provided
        if not messages:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})

            # Handle multimodal user content (text + images)
            if base64_images:
                content_blocks = []
                if user_message:
                    content_blocks.append({"type": "text", "text": user_message})
                for img_b64 in base64_images:
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    })
                messages.append({"role": "user", "content": content_blocks})
            else:
                if user_message:
                    messages.append({"role": "user", "content": user_message})

        payload = {
            "model": llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        logger.info(f"Streaming request to {llm_provider}: model={llm_model}")
        session = None
        try:
            # Create session with custom connector for Windows
            connector = aiohttp.TCPConnector(force_close=True) if sys.platform == 'win32' else None
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout),
                connector=connector
            )
            
            async with session.post(api_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for raw_line in response.content:
                        if not raw_line:
                            continue
                        line = raw_line.decode("utf-8").strip()
                        if not line:
                            continue
                        # OpenAI streams multiple lines that may begin with 'data:'; join those if needed.
                        if line.startswith("data: "):
                            data_str = line[len("data: ") :].strip()
                        else:
                            data_str = line
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            choices = data_json.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content_piece = delta.get("content")
                                if content_piece:
                                    yield content_piece
                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode JSON line from OpenAI stream: {data_str}")
        except Exception as e:
            logger.error(f"Error during OpenAI streaming: {e}", exc_info=True)
            yield f"[OpenAI streaming error: {e}]"
        finally:
            # Ensure session is properly closed
            if session and not session.closed:
                await session.close()
                # Small delay to allow cleanup on Windows
                if sys.platform == 'win32':
                    await asyncio.sleep(0.1)
        return

    if provider_lower == "gemini":
        if not llm_api_key:
            logger.error("Gemini streaming requested but no API key supplied.")
            yield "[Gemini Error: API key missing]"
            return
        
        async for chunk in send_gemini_request_stream(
            api_key=llm_api_key,
            model=llm_model,
            system_message=system_message,
            user_message=user_message,
            messages=messages,
            base64_images=base64_images,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
        ):
            yield chunk
        return

    if provider_lower == "ollama":
        # --- Ollama Specific Streaming Logic ---
        if not ensure_ollama_server(base_ip, port):
            logger.error("Ollama daemon unavailable and could not be started – aborting stream.")
            yield "[Error: Ollama daemon unavailable]"
            return

        # Ensure requested model is present locally (will pull if missing)
        ensure_ollama_model(llm_model, base_ip, port)

        # Ollama expects plain base64 strings without the 'data:image/...' prefix
        if base64_images:
            base64_images = [
                img.split("base64,")[1] if isinstance(img, str) and "base64," in img else img
                for img in base64_images
            ]

        # Decide which Ollama endpoint to use:
        #   • /api/generate  – fast text-only streaming (no image support)
        #   • /api/chat      – full chat/completions with image support
        # We switch to /api/chat automatically if the caller supplied base64-encoded images
        # so that vision models (e.g. qwen-vl, llava) receive the image bytes.

        use_chat_endpoint = bool(base64_images)  # True if we have images to send

        url = (
            f"http://{base_ip}:{port}/api/chat" if use_chat_endpoint else f"http://{base_ip}:{port}/api/generate"
        )

        headers = {"Content-Type": "application/json"}
        if llm_api_key: # Ollama doesn't typically use API keys this way, but include for consistency
            headers["Authorization"] = f"Bearer {llm_api_key}"

        # ------------------------------------------------------------------
        # Build request payloads
        # ------------------------------------------------------------------

        if use_chat_endpoint:
            # --------------------------------------------------------------
            #  /api/chat  (supports multimodal, messages array)
            # --------------------------------------------------------------
            if not messages:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})

            # Always append the user message at the end so that vision models
            # get the most recent prompt + images in a single message.
            user_msg: dict[str, Any] = {"role": "user", "content": user_message or ""}
            if base64_images:
                user_msg["images"] = base64_images  # Ollama expects a list key called "images"
            messages.append(user_msg)

            payload = {
                "model": llm_model,
                "messages": messages,
                "stream": True,
                "options": {
                    "seed": seed,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_k": top_k,
                    "top_p": top_p,
                    "repeat_penalty": repeat_penalty,
                    "stop": stop,
                },
            }
        else:
            # --------------------------------------------------------------
            #  /api/generate  (text-only)
            # --------------------------------------------------------------
            # Construct messages list if not provided directly (for context)
            if not messages:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                if user_message:
                    messages.append({"role": "user", "content": user_message})

            payload = {
                "model": llm_model,
                "prompt": user_message,
                "system": system_message if system_message else None,
                "stream": True,
                "options": {
                    "seed": seed,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_k": top_k,
                    "top_p": top_p,
                    "repeat_penalty": repeat_penalty,
                    "stop": stop,
                },
            }
            # Clean up None values Ollama might not like
            if not system_message:
                del payload["system"]

        # Remove None values from options dict
        if "options" in payload:
            payload["options"] = {k: v for k, v in payload["options"].items() if v is not None}

        logger.info(f"Streaming request to Ollama: {url} with payload: {{'model': '{llm_model}', 'stream': True, 'use_chat': {use_chat_endpoint}}}")

        try:
            # Use a single session if possible, manage timeouts
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                    # Process the streaming response line by line.  The JSON schema differs slightly
                    # between /api/generate and /api/chat, so we branch inside the loop.
                    async for line in response.content:
                        if not line:
                            continue
                        try:
                            decoded_line = line.decode("utf-8").strip()
                            if not decoded_line:
                                continue

                            data = json.loads(decoded_line)

                            if use_chat_endpoint:
                                # Chat endpoint returns {"message": {"content": "..."}, "done": bool}
                                msg = data.get("message", {})
                                chunk = msg.get("content", "")
                                is_done = data.get("done", False)
                            else:
                                # Generate endpoint returns {"response": "...", "done": bool}
                                chunk = data.get("response", "")
                                is_done = data.get("done", False)

                            if chunk:
                                yield chunk

                            if is_done:
                                logger.info("Ollama stream finished.")
                                break
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Could not decode JSON line from Ollama stream: {line.decode('utf-8', errors='ignore')}"
                            )
                        except Exception as e:
                            logger.error(f"Error processing Ollama stream line: {e}", exc_info=True)
                            yield f"[Error processing stream: {e}]"
                            break  # Stop streaming on error

        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection error to {url}: {e}")
            yield f"[Connection Error: {e}]"
        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error {e.status} from {url}: {e.message}")
            # Attempt to read error details from response body
            try:
                error_body = await e.response.text() if hasattr(e, 'response') else 'No details'
                logger.error(f"Error Body: {error_body}")
                yield f"[HTTP Error {e.status}: {e.message} - {error_body[:100]}]"
            except:
                yield f"[HTTP Error {e.status}: {e.message}]"
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {timeout} seconds to {url}")
            yield f"[Timeout Error]"
        except ConnectionResetError as e:
            logger.warning(f"Connection reset by peer: {e}")
            yield f"[Connection Reset: The server closed the connection]"
        except OSError as e:
            if e.errno == 10054:  # Windows specific: connection forcibly closed
                logger.warning("Connection forcibly closed by remote host")
                yield f"[Connection closed by server]"
            else:
                logger.error(f"OS Error during streaming: {e}")
                yield f"[Network Error: {e}]"
        except Exception as e:
            logger.error(f"An unexpected error occurred during streaming request: {e}", exc_info=True)
            yield f"[Unexpected Error: {e}]"

    if provider_lower == "groq":
        # --- Groq Specific Streaming Logic ---
        if not llm_api_key:
            logger.error("Groq streaming requested but no API key supplied.")
            # Fallback could be added here if desired, for now just yield error
            yield "[Groq Error: API key missing]"
            return

        groq_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {llm_api_key}",
            "Content-Type": "application/json",
        }

        # Handle multimodal user content (text + images)
        is_vision_model = "scout" in llm_model or "maverick" in llm_model
        images_to_send = base64_images
        if images_to_send and not is_vision_model:
            logger.warning("Groq stream: Model '%s' may not support images. Sending without.", llm_model)
            images_to_send = None
        elif images_to_send and len(images_to_send) > 5:
            logger.warning("Groq stream: Taking first 5 of %s images for vision model.", len(images_to_send))
            images_to_send = images_to_send[:5]

        # Build message list if not provided
        if not messages:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})

            if images_to_send:
                content_blocks = [{"type": "text", "text": user_message}]
                for img_b64 in images_to_send:
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    })
                messages.append({"role": "user", "content": content_blocks})
            else:
                if user_message:
                    messages.append({"role": "user", "content": user_message})

        payload = {
            "model": llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens, # Use correct Groq param
            "top_p": top_p,
            "stream": True,
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        logger.info(f"Streaming request to Groq: model={llm_model}")
        session = None
        try:
            # Create session with custom connector for Windows
            connector = aiohttp.TCPConnector(force_close=True) if sys.platform == 'win32' else None
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout),
                connector=connector
            )
            
            async with session.post(groq_url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    async for raw_line in response.content:
                        line = raw_line.decode("utf-8").strip()
                        if not line: continue
                        if line.startswith("data: "):
                            data_str = line[len("data: ") :].strip()
                        else:
                            data_str = line
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            choices = data_json.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content_piece = delta.get("content")
                                if content_piece:
                                    yield content_piece
                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode JSON line from Groq stream: {data_str}")
        except Exception as e:
            logger.error(f"Error during Groq streaming: {e}", exc_info=True)
            yield f"[Groq streaming error: {e}]"
        finally:
            # Ensure session is properly closed
            if session and not session.closed:
                await session.close()
                # Small delay to allow cleanup on Windows
                if sys.platform == 'win32':
                    await asyncio.sleep(0.1)
        return

    # Existing fallback logic for other providers
    if provider_lower not in ["ollama", "openai", "openrouter", "transformers", "hf", "local", "groq", "gemini"]:
        logger.warning(f"Streaming not implemented for provider '{llm_provider}'. Falling back to non-streaming.")
        try:
            full_response_data = await send_request(
                llm_provider=llm_provider, base_ip=base_ip, port=port, images=base64_images, llm_model=llm_model,
                system_message=system_message, user_message=user_message, messages=messages, seed=seed,
                temperature=temperature, max_tokens=max_tokens, random=random, top_k=top_k, top_p=top_p,
                repeat_penalty=repeat_penalty, stop=stop, keep_alive=keep_alive, llm_api_key=llm_api_key
            )
            if isinstance(full_response_data, dict):
                 if "choices" in full_response_data and full_response_data["choices"]:
                     message = full_response_data["choices"][0].get("message", {})
                     content = message.get("content", "")
                     if content: yield content
                 elif "response" in full_response_data:
                     if full_response_data["response"]: yield full_response_data["response"]
                 elif "candidates" in full_response_data and full_response_data.get("candidates"):
                     try:
                         content = full_response_data["candidates"][0]["content"]["parts"][0]["text"]
                         if content: yield content
                     except (KeyError, IndexError, TypeError):
                         logger.warning(f"Could not parse Gemini response format: {full_response_data}")
                 else:
                    logger.error(f"Unexpected non-streaming response format: {full_response_data}")
            elif isinstance(full_response_data, str):
                 if full_response_data: yield full_response_data
            else:
                 logger.error(f"Unexpected non-streaming response type: {type(full_response_data)}")

        except Exception as e:
            logger.error(f"Error in fallback non-streaming request for {llm_provider}: {e}", exc_info=True)
            yield f"[Error: {e}]"
        return # Stop generation after yielding the fallback response

def _remove_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks from text, including the tags themselves."""
    import re
    # Pattern to match <think>...</think> or ◁think▷...◁/think▷
    pattern = r'<think>.*?</think>|◁think▷.*?◁/think▷'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    # Clean up any extra whitespace/newlines left behind
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Replace multiple newlines with double
    return cleaned.strip()

# --- Original Node (for reference or non-streaming use) ---
class LLMToolkitTextGenerator:
    DEFAULT_PROVIDER = "openai"
    
    DEFAULT_MODEL: str = "gpt-4o-mini"

    MODEL_LIST: List[str] = [DEFAULT_MODEL]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": False, "default": "Write a short story about a robot learning to paint."}),
                "hide_thinking": ("BOOLEAN", {"default": True, "tooltip": "Hide model thinking process (content between <think> tags)"})
            },
            "optional": {
                "context": ("*", {}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 65536}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 500}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "-1 = random seed"})
            },
            "hidden": {
                "llm_model": ("STRING", {"default": cls.DEFAULT_MODEL})
            }
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("context", "text")
    FUNCTION = "generate"
    CATEGORY = "🔗llm_toolkit/generators"
    OUTPUT_NODE = True # Keeps the text widget for non-streaming version

    def generate(self, prompt, hide_thinking, context=None, temperature=0.7, max_tokens=1024, top_p=0.9, top_k=40, seed=-1, llm_model=None):
        # ... (original generate logic using run_async(send_request(...))) ...
        # This function remains mostly the same as the user provided,
        # calling the original non-streaming send_request.
        # We'll copy the parameter processing logic from the streaming version
        # for consistency, but it will call the non-streaming send_request.
        try:
            # Base parameter defaults
            params = {
                "llm_provider": self.DEFAULT_PROVIDER,
                "llm_model": llm_model or self.DEFAULT_MODEL,
                "system_message": "You are a helpful, creative, and concise assistant.",
                "user_message": prompt,
                "base_ip": "localhost",
                "port": "11434",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed if seed != -1 else None,
                "repeat_penalty": 1.1,
                "stop": None,
                "keep_alive": True,
                "messages": [],
            }

            provider_config = None
            if context is not None:
                if isinstance(context, dict) and "provider_name" in context:
                    provider_config = context
                elif isinstance(context, dict) and "provider_config" in context:
                    provider_config = context["provider_config"]

            if provider_config and isinstance(provider_config, dict):
                if "provider_name" in provider_config: params["llm_provider"] = provider_config["provider_name"]
                if "api_key" in provider_config: params["llm_api_key"] = provider_config["api_key"]
                if "base_ip" in provider_config: params["base_ip"] = provider_config["base_ip"]
                if "port" in provider_config: params["port"] = provider_config["port"]
                for key in provider_config:
                    if key not in ["provider_name", "llm_model", "api_key", "base_ip", "port", "user_prompt"]:
                        params[key] = provider_config[key]
                if "user_prompt" in provider_config: params["user_message"] = provider_config["user_prompt"]
                provider_model = provider_config.get("llm_model", "")
                if provider_model:
                    params["llm_model"] = provider_model
                else:
                    params["llm_model"] = "" # Will be handled below

            if params.get("llm_provider"):
                provider = str(params["llm_provider"]).lower()
                if not params.get("llm_model"):
                    PROVIDER_DEFAULTS = {"openai": "gpt-4o-mini", "anthropic": "claude-3-opus-20240229"}
                    fallback = PROVIDER_DEFAULTS.get(provider)
                    if fallback: params["llm_model"] = fallback

                # --- Groq reasoning-format handling ---
                if provider == "groq":
                    # Hide thinking via API when requested to avoid post-processing.
                    params["reasoning_format"] = "hidden" if hide_thinking else "raw"

            # Auto-fetch API key for OpenAI if missing/placeholder
            if provider == "openai" and (not params.get("llm_api_key") or params["llm_api_key"] in {"", "1234", None}):
                try:
                    params["llm_api_key"] = get_api_key("OPENAI_API_KEY", "openai")
                    logger.info("generate: Retrieved OpenAI API key via get_api_key helper.")
                except ValueError as _e:
                    logger.warning(f"generate: get_api_key failed – {_e}")

            # --- Prompt-Manager support ---------------------------------------------------
            # If upstream PromptManager attached a prompt_config dict, use the data
            prompt_cfg = None
            if context is not None and isinstance(context, dict):
                prompt_cfg = context.get("prompt_config")

            if prompt_cfg and isinstance(prompt_cfg, dict):
                # Override user prompt text if provided
                if prompt_cfg.get("text"):
                    params["user_message"] = prompt_cfg["text"]

                # Collect images and video frames (already in base64 from PromptManager)
                imgs = []
                if prompt_cfg.get("image_base64"):
                    img_val = prompt_cfg["image_base64"]
                    if isinstance(img_val, str):
                        imgs.extend([img_val])
                    elif isinstance(img_val, list):
                        imgs.extend(img_val)

                # Add extracted video frames
                if prompt_cfg.get("video_frames_base64"):
                    vid_frames = prompt_cfg["video_frames_base64"]
                    if isinstance(vid_frames, list):
                        imgs.extend(vid_frames)
                    else:
                        imgs.append(vid_frames)

                params["images"] = imgs if imgs else None

                # Pass through file paths and URLs for APIs that support them
                if prompt_cfg.get("file_paths"):
                    params["file_paths"] = prompt_cfg["file_paths"]
                    logger.debug(f"PromptManager: Passing file_paths to API (some APIs process videos/PDFs directly).")

                if prompt_cfg.get("urls"):
                    params["urls"] = prompt_cfg["urls"]
                    logger.debug(f"PromptManager: Passing URLs to API.")

                # Audio path remains separate
                if prompt_cfg.get("audio_path"):
                    params["audio_path"] = prompt_cfg["audio_path"]
            else:
                params["images"] = None

            # --- New: If file_paths contain local video files, extract a few frames ---
            try:
                file_paths_value = params.get("file_paths")
                candidate_paths: List[str] = []
                if isinstance(file_paths_value, str) and file_paths_value.strip():
                    candidate_paths = [file_paths_value.strip()]
                elif isinstance(file_paths_value, list):
                    candidate_paths = [p for p in file_paths_value if isinstance(p, str) and p.strip()]

                extracted_from_files: List[str] = []
                for p in candidate_paths:
                    if _is_video_file(p) and os.path.isfile(p):
                        extracted_from_files.extend(_extract_video_file_frames_as_b64(p))

                if extracted_from_files:
                    if params.get("images") is None:
                        params["images"] = []
                    if not isinstance(params["images"], list):
                        params["images"] = [params["images"]]  # normalize
                    # cap total images to 8 to avoid huge payloads
                    remaining = max(0, 8 - len(params["images"]))
                    params["images"].extend(extracted_from_files[:remaining])
                    logger.info("Added %d frame(s) extracted from video file paths to images payload.", min(len(extracted_from_files), remaining))
            except Exception as e:
                logger.warning("Failed to process video file paths for frame extraction: %s", e, exc_info=True)

            log_params = _sanitize_params_for_log(params)
            logger.info(f"[Non-Streaming] Making LLM request with params: {log_params}")

            try:
                # --- CALL NON-STREAMING VERSION ---
                response_data = run_async(
                    send_request( # Original non-streaming call
                        llm_provider=params["llm_provider"],
                        base_ip=params.get("base_ip", "localhost"),
                        port=params.get("port", "11434"),
                        images=params.get("images"),
                        llm_model=params["llm_model"],
                        system_message=params["system_message"],
                        user_message=params["user_message"],
                        messages=params["messages"],
                        seed=params.get("seed"),
                        temperature=params["temperature"],
                        max_tokens=params["max_tokens"],
                        random=params.get("random", False),
                        top_k=params["top_k"],
                        top_p=params["top_p"],
                        repeat_penalty=params["repeat_penalty"],
                        stop=params.get("stop"),
                        keep_alive=params.get("keep_alive", True),
                        llm_api_key=params.get("llm_api_key"),
                        reasoning_format=params.get("reasoning_format") # Pass reasoning_format
                    )
                )
                # --- END NON-STREAMING CALL ---
            except Exception as e:
                 logger.error(f"Error in non-streaming send_request call: {e}", exc_info=True)
                 response_data = {"choices": [{"message": {"content": f"Error calling send_request: {str(e)}"}}]}

            if response_data is None: 
                content = "Error: Received None response"
            elif isinstance(response_data, dict):
                if "choices" in response_data and response_data["choices"]:
                    message = response_data["choices"][0].get("message", {})
                    content = message.get("content", "")
                    if content is None: 
                        content = "Error: Null content in response"
                elif "response" in response_data: 
                    content = response_data["response"]
                else: 
                    content = f"Error: Unexpected format: {str(response_data)}"
            elif isinstance(response_data, str): 
                content = response_data
            else: 
                content = f"Error: Unexpected type: {type(response_data)}"

            # Apply thinking tag removal if requested
            if hide_thinking and content:
                content = _remove_thinking_tags(content)

            if context is not None and isinstance(context, dict):
                context_out = context.copy()
                context_out["llm_response"] = content
                context_out["llm_raw_response"] = response_data
            else:
                context_out = {"llm_response": content, "llm_raw_response": response_data, "passthrough_data": context}

            payload = ContextPayload(content, context_out)
            return {"ui": {"string": [content]}, "result": (payload, str(payload))}

        except Exception as e:
            error_message = f"Error generating text: {str(e)}"
            logger.error(error_message, exc_info=True)
            error_output = {"error": error_message, "original_input": context}
            payload = ContextPayload(error_message, error_output)
            return {"ui": {"string": [error_message]}, "result": (payload, str(payload))}

# --- Helper to avoid dumping large base64 blobs to INFO log -----------------
def _sanitize_params_for_log(d: dict) -> dict:
    """Return a shallow copy with huge fields replaced by short placeholders."""
    out = {}
    for k, v in d.items():
        # Hide large binary blobs (images, video frames)
        if k in {"images", "video_frames", "base64_images"} and v:
            # Replace with concise placeholder
            if isinstance(v, (list, tuple)):
                out[k] = f"[{len(v)} item(s) base64 omitted]"
            elif isinstance(v, str):
                out[k] = "[base64 string omitted]"
            else:
                out[k] = "[data omitted]"
            continue

        # Mask any values that look like API keys so we never log them in full
        key_lc = k.lower()
        if "api_key" in key_lc or key_lc.endswith("_key") or key_lc.endswith("key"):
            if isinstance(v, str) and v:
                # Keep first 5 chars for debugging, mask the rest
                masked = v[:5] + "…" if len(v) > 5 else "***"
                out[k] = masked
            else:
                out[k] = "[key hidden]"
            continue

        # Default passthrough
        else:
            out[k] = v
    return out

# --- NEW STREAMING NODE ---
class LLMToolkitTextGeneratorStream:
    DEFAULT_PROVIDER = "openai"

    DEFAULT_MODEL: str = "gpt-4o-mini"

    MODEL_LIST: List[str] = [DEFAULT_MODEL]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": False, "default": "Write a detailed description of a futuristic city."}),
                "hide_thinking": ("BOOLEAN", {"default": True, "tooltip": "Hide model thinking process (content between <think> tags)"})
            },
            "optional": {
                "context": ("*", {}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 65536}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 500}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "-1 = random seed"})
            },
            "hidden": { # <-- Add hidden inputs
                "unique_id": "UNIQUE_ID",
                "llm_model": ("STRING", {"default": cls.DEFAULT_MODEL})
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("context", "text")
    FUNCTION = "generate_stream" # <-- Use new function name
    CATEGORY = "🔗llm_toolkit/generators"
    OUTPUT_NODE = True # Keep the JS widget logic

    def generate_stream(self, prompt, hide_thinking, unique_id, llm_model=None, context=None, **kwargs):
        """
        Generates text using the specified provider and streams the response back
        to the UI via websocket messages.  (synchronous wrapper)
        """
        # Wrap the previous async implementation inside an inner coroutine
        async def _async_generate():
            # Previous async body START
            if PromptServer is None:
                logger.error("PromptServer not available. Cannot stream.")
                error_msg = "Streaming requires PromptServer, which is not available."
                error_output = {"error": error_msg, "original_input": context}
                payload = ContextPayload(error_msg, error_output)
                return {"ui": {"string": [error_msg]}, "result": (payload, str(payload))}

            server = PromptServer.instance
            full_response_text = ""
            thinking_buffer = ""
            inside_thinking = False

            try:
                # --- Corrected Parameter Processing Logic ---
                # 1. Start with node defaults
                params = {
                    "llm_provider": self.DEFAULT_PROVIDER,
                    "llm_model": llm_model or self.DEFAULT_MODEL,
                    "system_message": "You are a helpful, creative, and concise assistant.",
                    "user_message": prompt,
                    "base_ip": "localhost", "port": "11434",
                    "temperature": kwargs.get('temperature', 0.7), "max_tokens": kwargs.get('max_tokens', 1024), "top_p": kwargs.get('top_p', 0.9), "top_k": kwargs.get('top_k', 40), "seed": kwargs.get('seed', -1),
                    "repeat_penalty": 1.1, "stop": None, "keep_alive": "5m",
                    "messages": [], "llm_api_key": None,
                }

                # 2. Apply general settings from the incoming context
                if context and isinstance(context, dict):
                    for key, value in context.items():
                        if key in params and value is not None:
                            params[key] = value

                # 3. Apply specific provider_config settings, which take precedence
                provider_config = None
                if context and isinstance(context, dict):
                    if "provider_config" in context and isinstance(context["provider_config"], dict):
                        provider_config = context["provider_config"]
                    elif "provider_name" in context: # Handle flat context from old nodes
                        provider_config = context
                
                if provider_config:
                    # First, update all matching keys from provider_config that also exist in params
                    params.update({k: v for k, v in provider_config.items() if k in params and v is not None})
                            
                    # Second, handle specific key name differences. This ensures provider_name is mapped correctly.
                    if "provider_name" in provider_config and provider_config["provider_name"]:
                        params["llm_provider"] = provider_config["provider_name"]
                    elif context and isinstance(context, dict) and context.get("provider_name"):
                        # Fallback to root-level key if nested provider_config didn't have it
                        params["llm_provider"] = context["provider_name"]

                    if "api_key" in provider_config and provider_config["api_key"]:
                        params["llm_api_key"] = provider_config["api_key"]
                    elif context and isinstance(context, dict) and context.get("api_key"):
                        params["llm_api_key"] = context["api_key"]

                    if "llm_model" in provider_config and provider_config["llm_model"]:
                        params["llm_model"] = provider_config["llm_model"]
                    elif context and isinstance(context, dict) and context.get("llm_model"):
                        params["llm_model"] = context["llm_model"]

                # 4. The user_message from context or provider_config can be used,
                #    but the node's prompt input has the final say.
                if context and isinstance(context, dict) and "user_prompt" in context and context["user_prompt"]:
                    params["user_message"] = context.get("user_prompt")
                if provider_config and "user_prompt" in provider_config and provider_config["user_prompt"]:
                     params["user_message"] = provider_config.get("user_prompt")
                if prompt: # Node input is final override, if provided
                    params["user_message"] = prompt

                # --- End Corrected Parameter Processing ---

                # Finalize model name fallback
                provider = "unknown"
                if params.get("llm_provider"):
                    provider = str(params["llm_provider"]).lower()
                    if not params.get("llm_model"):
                        PROVIDER_DEFAULTS = {"openai": "gpt-4o-mini", "anthropic": "claude-3-opus-20240229"}
                        fallback = PROVIDER_DEFAULTS.get(provider)
                        params["llm_model"] = fallback or self.DEFAULT_MODEL

                        # Groq reasoning format
                        if provider == "groq":
                            params["reasoning_format"] = "hidden" if hide_thinking else "raw"

                # Auto-fetch API key for OpenAI if missing/placeholder
                if provider == "openai" and (not params.get("llm_api_key") or params["llm_api_key"] in {"", "1234", None}):
                    try:
                        params["llm_api_key"] = get_api_key("OPENAI_API_KEY", "openai")
                        logger.info("generate_stream: Retrieved OpenAI API key via get_api_key helper.")
                    except ValueError as _e:
                        logger.warning(f"generate_stream: get_api_key failed – {_e}")

                # --- Prompt-Manager support ---------------------------------------------------
                # If upstream PromptManager attached a prompt_config dict, use the data
                prompt_cfg = None
                if context is not None and isinstance(context, dict):
                    prompt_cfg = context.get("prompt_config")

                if prompt_cfg and isinstance(prompt_cfg, dict):
                    # Override user prompt text if provided
                    if prompt_cfg.get("text"):
                        params["user_message"] = prompt_cfg["text"]

                    # Collect images and video frames (already in base64 from PromptManager)
                    imgs = []
                    if prompt_cfg.get("image_base64"):
                        img_val = prompt_cfg["image_base64"]
                        if isinstance(img_val, str):
                            imgs.extend([img_val])
                        elif isinstance(img_val, list):
                            imgs.extend(img_val)

                    # Add extracted video frames
                    if prompt_cfg.get("video_frames_base64"):
                        vid_frames = prompt_cfg["video_frames_base64"]
                        if isinstance(vid_frames, list):
                            imgs.extend(vid_frames)
                        else:
                            imgs.append(vid_frames)

                    params["images"] = imgs if imgs else None

                    # Pass through file paths and URLs for APIs that support them
                    if prompt_cfg.get("file_paths"):
                        params["file_paths"] = prompt_cfg["file_paths"]
                        logger.debug(f"PromptManager: Passing file_paths to API (some APIs process videos/PDFs directly).")

                    if prompt_cfg.get("urls"):
                        params["urls"] = prompt_cfg["urls"]
                        logger.debug(f"PromptManager: Passing URLs to API.")

                    # Audio path remains separate
                    if prompt_cfg.get("audio_path"):
                        params["audio_path"] = prompt_cfg["audio_path"]
                else:
                    params["images"] = None

                # --- New: If file_paths contain local video files, extract a few frames ---
                try:
                    file_paths_value = params.get("file_paths")
                    candidate_paths: List[str] = []
                    if isinstance(file_paths_value, str) and file_paths_value.strip():
                        candidate_paths = [file_paths_value.strip()]
                    elif isinstance(file_paths_value, list):
                        candidate_paths = [p for p in file_paths_value if isinstance(p, str) and p.strip()]

                    extracted_from_files: List[str] = []
                    for p in candidate_paths:
                        if _is_video_file(p) and os.path.isfile(p):
                            extracted_from_files.extend(_extract_video_file_frames_as_b64(p))

                    if extracted_from_files:
                        if params.get("images") is None:
                            params["images"] = []
                        if not isinstance(params["images"], list):
                            params["images"] = [params["images"]]  # normalize
                        # cap total images to 8 to avoid huge payloads
                        remaining = max(0, 8 - len(params["images"]))
                        params["images"].extend(extracted_from_files[:remaining])
                        logger.info(
                            "[Streaming] Added %d frame(s) extracted from video file paths to images payload.",
                            min(len(extracted_from_files), remaining),
                        )
                except Exception as e:
                    logger.warning(
                        "[Streaming] Failed to process video file paths for frame extraction: %s",
                        e,
                        exc_info=True,
                    )

                log_params = _sanitize_params_for_log(params)
                logger.info(
                    f"[Streaming] Initiating LLM stream with params: {log_params} for node {unique_id}"
                )

                # --- Send START message ---
                server.send_sync("llmtoolkit.stream.start", {"node": unique_id}, sid=server.client_id)

                # --- Initiate and process the stream ---
                stream_generator = send_request_stream(
                    llm_provider=params["llm_provider"],
                    base_ip=params.get("base_ip", "localhost"),
                    port=params.get("port", "11434"),
                    llm_model=params["llm_model"],
                    system_message=params["system_message"],
                    user_message=params["user_message"],
                    messages=params["messages"],
                    seed=params.get("seed"),
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    random=params.get("random", False),
                    top_k=params["top_k"],
                    top_p=params["top_p"],
                    repeat_penalty=params["repeat_penalty"],
                    stop=params.get("stop"),
                    keep_alive=params.get("keep_alive", True),
                    llm_api_key=params.get("llm_api_key"),
                    base64_images=params.get("images"),
                )

                async for chunk in stream_generator:
                    if chunk:
                        full_response_text += chunk
                        
                        # Handle thinking tag filtering for streaming
                        if hide_thinking:
                            # Track thinking state and buffer content
                            chunk_to_send = ""
                            i = 0
                            while i < len(chunk):
                                
                                if not inside_thinking:
                                    # Check for start of thinking tag
                                    if chunk[i:i+7] == '<think>':
                                        inside_thinking = True
                                        thinking_buffer = '<think>'
                                        i += 7
                                        continue
                                    elif chunk[i:i+9] == '◁think▷': # <-- Add Kimi-VL start tag
                                        inside_thinking = True
                                        thinking_buffer = '◁think▷'
                                        i += 9
                                        continue
                                    else:
                                        chunk_to_send += chunk[i]
                                else:
                                    # Inside thinking block, buffer until we find closing tag
                                    thinking_buffer += chunk[i]
                                    if thinking_buffer.endswith('</think>'):
                                        inside_thinking = False
                                        thinking_buffer = ""
                                        i += 1
                                        continue
                                    elif thinking_buffer.endswith('◁/think▷'): # <-- Add Kimi-VL end tag
                                        inside_thinking = False
                                        thinking_buffer = ""
                                        i += 1
                                        continue
                                i += 1
                            
                            # Only send non-thinking content
                            if chunk_to_send:
                                server.send_sync(
                                    "llmtoolkit.stream.chunk",
                                    {"node": unique_id, "text": chunk_to_send},
                                    sid=server.client_id,
                                )
                        else:
                            # Send all content if not hiding thinking
                            server.send_sync(
                                "llmtoolkit.stream.chunk",
                                {"node": unique_id, "text": chunk},
                                sid=server.client_id,
                            )
                    await asyncio.sleep(0.001)

                # Apply thinking tag removal to final text if requested
                final_text = full_response_text
                if hide_thinking:
                    final_text = _remove_thinking_tags(full_response_text)

                logger.info(
                    f"[Streaming] Finished for node {unique_id}. Total length: {len(final_text)}"
                )
                server.send_sync(
                    "llmtoolkit.stream.end",
                    {"node": unique_id, "final_text": final_text},
                    sid=server.client_id,
                )

                # --- Prepare final context output ---
                if context is not None and isinstance(context, dict):
                    context_out = context.copy()
                    context_out["llm_response"] = final_text
                    context_out["llm_raw_response"] = {
                        "status": "Streamed successfully",
                        "final_length": len(final_text),
                    }
                else:
                    context_out = {
                        "llm_response": final_text,
                        "llm_raw_response": {
                            "status": "Streamed successfully",
                            "final_length": len(final_text),
                        },
                        "passthrough_data": context,
                    }

                payload = ContextPayload(final_text, context_out)
                return {"ui": {"string": [final_text]}, "result": (payload, str(payload))}

            except Exception as e:
                error_message = f"Error during streaming generation: {str(e)}"
                logger.error(error_message, exc_info=True)
                if server and unique_id:
                    server.send_sync(
                        "llmtoolkit.stream.error",
                        {"node": unique_id, "error": error_message},
                        sid=server.client_id,
                    )
                error_output = {
                    "error": error_message,
                    "partial_response": full_response_text,
                    "original_input": context,
                }
                payload = ContextPayload(error_message, error_output)
                return {
                    "ui": {"string": [f"Error: {error_message}\nPartial: {full_response_text}"]},
                    "result": (payload, str(payload)),
                }
            # Previous async body END

        # Execute the inner coroutine and return its result synchronously
        return run_async(_async_generate())


# Node Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LLMToolkitTextGenerator": LLMToolkitTextGenerator, # Keep original
    "LLMToolkitTextGeneratorStream": LLMToolkitTextGeneratorStream # Add streaming version
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMToolkitTextGenerator": "Generate Text (🔗LLMToolkit)",
    "LLMToolkitTextGeneratorStream": "Generate Text Stream (🔗LLMToolkit)" # New display name
}