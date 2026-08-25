import time


class TerminalLatencyProfiler:
    """A lightweight ComfyUI custom node for logging execution timestamps

    and pass-through overhead directly to the server terminal stdout.
    """

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

    def profile_latency(self, passthrough_data, tag="Node Execution"):
        start_time = time.perf_counter()

        # Pass-through operation
        output_data = passthrough_data

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        current_time = time.strftime("%H:%M:%S")
        monotonic_stamp = time.perf_counter() * 1000.0

        print("=" * 50)
        print(f"[LATENCY LOG] {current_time}")
        print(f"Tag: {tag}")
        print(f"Monotonic Stamp: {monotonic_stamp:.2f} ms")
        print(f"Profiler Passthrough Overhead: {elapsed_ms:.4f} ms")
        print("=" * 50)

        return (output_data,)


NODE_CLASS_MAPPINGS = {
    "TerminalLatencyProfiler": TerminalLatencyProfiler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TerminalLatencyProfiler": "Terminal Latency Profiler",
}