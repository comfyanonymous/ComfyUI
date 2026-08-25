import re
import time


class TerminalLatencyProfiler:
    """A lightweight ComfyUI custom node for logging execution timestamps

    and pass-through overhead directly to the server terminal stdout.
    """

    MAX_TAG_LENGTH = 80
    _CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough_data": ("*",),
            },
            "optional": {
                "tag": ("STRING", {"default": "Node Execution"}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough_data",)
    FUNCTION = "profile_latency"
    CATEGORY = "utils/profiling"

    @classmethod
    def IS_CHANGED(cls, passthrough_data, tag="Node Execution"):
        """Force ComfyUI to re-execute this node on every queue submission

        rather than returning cached outputs.
        """
        return float("nan")

    @classmethod
    def _sanitize_tag(cls, tag: str) -> str:
        """Strip control/newline characters and cap length so a

        workflow-controlled label can't inject extra terminal lines
        or flood stdout.
        """
        clean = cls._CONTROL_CHARS_RE.sub("", tag).strip()
        if len(clean) > cls.MAX_TAG_LENGTH:
            clean = clean[: cls.MAX_TAG_LENGTH - 1].rstrip() + "…"
        return clean or "Node Execution"

    def profile_latency(self, passthrough_data, tag="Node Execution"):
        start_time = time.perf_counter()

        # Pass-through operation
        output_data = passthrough_data

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        current_time = time.strftime("%H:%M:%S")
        monotonic_stamp = time.perf_counter() * 1000.0
        safe_tag = self._sanitize_tag(str(tag))

        print(
            f"[LATENCY LOG] {current_time} | Tag: {safe_tag} | "
            f"Stamp: {monotonic_stamp:.2f}ms | Overhead: {elapsed_ms:.4f}ms"
        )

        return (output_data,)


NODE_CLASS_MAPPINGS = {
    "TerminalLatencyProfiler": TerminalLatencyProfiler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TerminalLatencyProfiler": "Terminal Latency Profiler",
}