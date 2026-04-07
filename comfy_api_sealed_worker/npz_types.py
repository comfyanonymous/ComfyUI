from __future__ import annotations

import os


class NPZ:
    """Ordered collection of NPZ file payloads.

    Each entry in ``frames`` is a complete compressed ``.npz`` file stored
    as raw bytes (produced by ``numpy.savez_compressed`` into a BytesIO).
    ``save_to`` writes numbered files into a directory.
    """

    def __init__(self, frames: list[bytes]) -> None:
        self.frames = frames

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def save_to(self, directory: str, prefix: str = "frame") -> str:
        os.makedirs(directory, exist_ok=True)
        for i, frame_bytes in enumerate(self.frames):
            path = os.path.join(directory, f"{prefix}_{i:06d}.npz")
            with open(path, "wb") as f:
                f.write(frame_bytes)
        return directory
