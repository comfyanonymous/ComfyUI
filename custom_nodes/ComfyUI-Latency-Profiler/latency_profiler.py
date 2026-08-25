import time

class ComfyLatencyProfiler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "passthrough_data": ("*",),  # Accepts any type (IMAGE, LATENT, MODEL, etc.)
            },
            "optional": {
                "tag": ("STRING", {"default": "Node Execution"}),
            }
        }

    RETURN_TYPES = ("*", "STRING", "FLOAT")
    RETURN_NAMES = ("passthrough_data", "latency_report", "latency_ms")
    FUNCTION = "profile_latency"
    CATEGORY = "Performance / Profiling"

    def profile_latency(self, passthrough_data, tag="Node Execution"):
        start_time = time.perf_counter()

        # Pass data through cleanly without modifying it
        output_data = passthrough_data

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        report = f"⚡ [{tag}] Latency: {elapsed_ms:.2f} ms"

        print(f"\n[ComfyUI-Latency-Profiler] {report}")

        return (output_data, report, elapsed_ms)