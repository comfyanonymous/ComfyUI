import time

class TerminalLatencyProfiler:
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
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output_signal",)
    FUNCTION = "log_terminal_latency"
    CATEGORY = "Performance"
    OUTPUT_NODE = True  # Ensures execution even if outputs aren't connected

    @classmethod
    def IS_CHANGED(s, input_signal, step_label="Checkpoint / Node"):
        # Force execution on every queue submission by bypassing node caching
        return float("nan")

    def log_terminal_latency(self, input_signal, step_label="Checkpoint / Node"):
        start_time = time.perf_counter()
        
        current_time = time.strftime("%H:%M:%S")
        timestamp_ms = start_time * 1000.0

        # Calculate lightweight pass-through execution overhead
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Output formatted metrics to terminal
        print("\n" + "=" * 50)
        print(f"[LATENCY LOG] {current_time}")
        print(f"Label: {step_label}")
        print(f"Monotonic Stamp: {timestamp_ms:.2f} ms")
        print(f"Pass-Through Overhead: {execution_time_ms:.4f} ms")
        print("=" * 50 + "\n")

        return (input_signal,)

NODE_CLASS_MAPPINGS = {"TerminalLatencyProfiler": TerminalLatencyProfiler}
NODE_DISPLAY_NAME_MAPPINGS = {"TerminalLatencyProfiler": "Terminal Latency Profiler"}