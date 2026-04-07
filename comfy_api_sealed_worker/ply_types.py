from __future__ import annotations

import numpy as np


class PLY:
    """Point cloud payload for PLY file output.

    Supports two schemas:
    - Pointcloud: xyz positions with optional colors, confidence, view_id (ASCII format)
    - Gaussian: raw binary PLY data built by producer nodes using plyfile (binary format)

    When ``raw_data`` is provided, the object acts as an opaque binary PLY
    carrier and ``save_to`` writes the bytes directly.
    """

    def __init__(
        self,
        points: np.ndarray | None = None,
        colors: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
        view_id: np.ndarray | None = None,
        raw_data: bytes | None = None,
    ) -> None:
        self.raw_data = raw_data
        if raw_data is not None:
            self.points = None
            self.colors = None
            self.confidence = None
            self.view_id = None
            return
        if points is None:
            raise ValueError("Either points or raw_data must be provided")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {points.shape}")
        self.points = np.ascontiguousarray(points, dtype=np.float32)
        self.colors = np.ascontiguousarray(colors, dtype=np.float32) if colors is not None else None
        self.confidence = np.ascontiguousarray(confidence, dtype=np.float32) if confidence is not None else None
        self.view_id = np.ascontiguousarray(view_id, dtype=np.int32) if view_id is not None else None

    @property
    def is_gaussian(self) -> bool:
        return self.raw_data is not None

    @property
    def num_points(self) -> int:
        if self.points is not None:
            return self.points.shape[0]
        return 0

    @staticmethod
    def _to_numpy(arr, dtype):
        if arr is None:
            return None
        if hasattr(arr, "numpy"):
            arr = arr.cpu().numpy() if hasattr(arr, "cpu") else arr.numpy()
        return np.ascontiguousarray(arr, dtype=dtype)

    def save_to(self, path: str) -> str:
        if self.raw_data is not None:
            with open(path, "wb") as f:
                f.write(self.raw_data)
            return path
        self.points = self._to_numpy(self.points, np.float32)
        self.colors = self._to_numpy(self.colors, np.float32)
        self.confidence = self._to_numpy(self.confidence, np.float32)
        self.view_id = self._to_numpy(self.view_id, np.int32)
        N = self.num_points
        header_lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {N}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if self.colors is not None:
            header_lines += ["property uchar red", "property uchar green", "property uchar blue"]
        if self.confidence is not None:
            header_lines.append("property float confidence")
        if self.view_id is not None:
            header_lines.append("property int view_id")
        header_lines.append("end_header")

        with open(path, "w") as f:
            f.write("\n".join(header_lines) + "\n")
            for i in range(N):
                parts = [f"{self.points[i, 0]} {self.points[i, 1]} {self.points[i, 2]}"]
                if self.colors is not None:
                    r, g, b = (self.colors[i] * 255).clip(0, 255).astype(np.uint8)
                    parts.append(f"{r} {g} {b}")
                if self.confidence is not None:
                    parts.append(f"{self.confidence[i]}")
                if self.view_id is not None:
                    parts.append(f"{int(self.view_id[i])}")
                f.write(" ".join(parts) + "\n")
        return path
