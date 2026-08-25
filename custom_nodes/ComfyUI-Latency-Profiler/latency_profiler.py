import sys
import time
import gc
import logging
import tracemalloc
import unittest
from dataclasses import dataclass
from typing import Any, Tuple, Dict, Optional

# Set up module logger
logger = logging.getLogger("ComfyUI.LatencyProfiler")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class PerformanceMetrics:
    step_label: str
    wall_time_str: str
    monotonic_stamp_ms: float
    elapsed_since_last_ms: Optional[float]
    memory_peak_mb: float


class TerminalLatencyProfiler:
    _last_timestamp: Optional[float] = None

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "input_signal": ("*",),
            },
            "optional": {
                "step_label": ("STRING", {"default": "Checkpoint / Node"}),
                "reset_timer": ("BOOLEAN", {"default": False}),
                "track_memory": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output_signal",)
    FUNCTION = "log_terminal_latency"
    CATEGORY = "Performance"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, input_signal: Any, step_label: str = "Checkpoint / Node", reset_timer: bool = False, track_memory: bool = True) -> float:
        # Force ComfyUI to bypass node execution cache on every queue run
        return float("nan")

    def log_terminal_latency(
        self, 
        input_signal: Any, 
        step_label: str = "Checkpoint / Node", 
        reset_timer: bool = False,
        track_memory: bool = True
    ) -> Tuple[Any]:
        
        current_perf = time.perf_counter()
        current_time_str = time.strftime("%H:%M:%S")
        timestamp_ms = current_perf * 1000.0

        # Optional timer reset for multi-stage workflow benchmarking
        if reset_timer:
            TerminalLatencyProfiler._last_timestamp = None

        # Calculate interval since last profiler marker in graph
        elapsed_ms: Optional[float] = None
        elapsed_str = "N/A (First Marker / Reset)"
        
        if TerminalLatencyProfiler._last_timestamp is not None:
            elapsed_ms = (current_perf - TerminalLatencyProfiler._last_timestamp) * 1000.0
            elapsed_str = f"{elapsed_ms:.2f} ms"

        # Track Python process heap allocations if enabled
        memory_peak_mb = 0.0
        if track_memory:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            _, peak = tracemalloc.get_traced_memory()
            memory_peak_mb = peak / (1024 * 1024)

        # Structure metrics payload
        metrics = PerformanceMetrics(
            step_label=step_label,
            wall_time_str=current_time_str,
            monotonic_stamp_ms=timestamp_ms,
            elapsed_since_last_ms=elapsed_ms,
            memory_peak_mb=memory_peak_mb
        )

        # Update last execution marker timestamp
        TerminalLatencyProfiler._last_timestamp = current_perf

        # Format console log
        self._output_formatted_log(metrics, elapsed_str, track_memory)

        return (input_signal,)

    def _output_formatted_log(self, metrics: PerformanceMetrics, elapsed_str: str, track_memory: bool) -> None:
        log_lines = [
            "",
            "=" * 50,
            f"[LATENCY LOG] {metrics.wall_time_str}",
            f"Label: {metrics.step_label}",
            f"Monotonic Stamp: {metrics.monotonic_stamp_ms:.2f} ms",
            f"Elapsed Since Last Marker: {elapsed_str}"
        ]
        
        if track_memory:
            log_lines.append(f"Peak Traced Memory: {metrics.memory_peak_mb:.2f} MB")
            
        log_lines.extend(["=" * 50, ""])
        
        print("\n".join(log_lines), flush=True)


NODE_CLASS_MAPPINGS = {"TerminalLatencyProfiler": TerminalLatencyProfiler}
NODE_DISPLAY_NAME_MAPPINGS = {"TerminalLatencyProfiler": "Terminal Latency Profiler"}


# ============================================================================
# Unit Tests (Run directly via `python latency_profiler.py` for maintenance)
# ============================================================================

class TestTerminalLatencyProfiler(unittest.TestCase):
    def setUp(self) -> None:
        TerminalLatencyProfiler._last_timestamp = None

    def test_cache_bypass(self) -> None:
        # Verify IS_CHANGED returns NaN to prevent ComfyUI caching
        import math
        self.assertTrue(math.isnan(TerminalLatencyProfiler.IS_CHANGED(None)))

    def test_interval_measurement(self) -> None:
        profiler = TerminalLatencyProfiler()
        
        # First node execution marker
        profiler.log_terminal_latency("signal_a", step_label="Step 1")
        
        # Simulate workflow execution delay
        time.sleep(0.05)
        
        # Second node execution marker
        start = time.perf_counter()
        profiler.log_terminal_latency("signal_b", step_label="Step 2")
        elapsed = (time.perf_counter() - start) * 1000.0
        
        self.assertIsNotNone(TerminalLatencyProfiler._last_timestamp)


if __name__ == "__main__":
    logger.info("Executing self-test suite for TerminalLatencyProfiler...")
    unittest.main()