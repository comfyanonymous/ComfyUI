import time

class TerminalLatencyProfiler:
    _last_timestamp = None

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_signal": ("*",),
            },
            "optional": {
                "step_label": ("STRING", {"default": "Checkpoint / Node"}),
                "reset_timer": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output_signal",)
    FUNCTION = "log_terminal_latency"
    CATEGORY = "Performance"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, input_signal, step_label="Checkpoint / Node", reset_timer=False):
        # Prevent ComfyUI from caching outputs across identical workflow submissions
        return float("nan")

    def log_terminal_latency(self, input_signal, step_label="Checkpoint / Node", reset_timer=False):
        current_perf = time.perf_counter()
        current_time_str = time.strftime("%H:%M:%S")
        timestamp_ms = current_perf * 1000.0

        if reset_timer:
            TerminalLatencyProfiler._last_timestamp = None

        # Calculate elapsed time since the previous profiler marker in the graph
        elapsed_str = "N/A (First Marker)"
        if TerminalLatencyProfiler._last_timestamp is not None:
            elapsed_ms = (current_perf - TerminalLatencyProfiler._last_timestamp) * 1000.0
            elapsed_str = f"{elapsed_ms:.2f} ms"

        # Update global marker timestamp
        TerminalLatencyProfiler._last_timestamp = current_perf

        print("\n" + "=" * 50)
        print(f"[LATENCY LOG] {current_time_str}")
        print(f"Label: {step_label}")
        print(f"Monotonic Stamp: {timestamp_ms:.2f} ms")
        print(f"Elapsed Since Last Marker: {elapsed_str}")
        print("=" * 50 + "\n", flush=True)

        return (input_signal,)

NODE_CLASS_MAPPINGS = {"TerminalLatencyProfiler": TerminalLatencyProfiler}
NODE_DISPLAY_NAME_MAPPINGS = {"TerminalLatencyProfiler": "Terminal Latency Profiler"}