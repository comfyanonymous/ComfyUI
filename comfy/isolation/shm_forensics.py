# pylint: disable=consider-using-from-import,import-outside-toplevel
from __future__ import annotations

import atexit
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Set

LOG_PREFIX = "]["
logger = logging.getLogger(__name__)


def _shm_debug_enabled() -> bool:
    return os.environ.get("COMFY_ISO_SHM_DEBUG") == "1"


class _SHMForensicsTracker:
    def __init__(self) -> None:
        self._started = False
        self._tracked_files: Set[str] = set()
        self._current_model_context: Dict[str, str] = {
            "id": "unknown",
            "name": "unknown",
            "hash": "????",
        }

    @staticmethod
    def _snapshot_shm() -> Set[str]:
        shm_path = Path("/dev/shm")
        if not shm_path.exists():
            return set()
        return {f.name for f in shm_path.glob("torch_*")}

    def start(self) -> None:
        if self._started or not _shm_debug_enabled():
            return
        self._tracked_files = self._snapshot_shm()
        self._started = True
        logger.debug(
            "%s SHM:forensics_enabled tracked=%d", LOG_PREFIX, len(self._tracked_files)
        )

    def stop(self) -> None:
        if not self._started:
            return
        self.scan("shutdown", refresh_model_context=True)
        self._started = False
        logger.debug("%s SHM:forensics_disabled", LOG_PREFIX)

    def _compute_model_hash(self, model_patcher: Any) -> str:
        try:
            model_instance_id = getattr(model_patcher, "_instance_id", None)
            if model_instance_id is not None:
                model_id_text = str(model_instance_id)
                return model_id_text[-4:] if len(model_id_text) >= 4 else model_id_text

            import torch

            real_model = (
                model_patcher.model
                if hasattr(model_patcher, "model")
                else model_patcher
            )
            tensor = None
            if hasattr(real_model, "parameters"):
                for p in real_model.parameters():
                    if torch.is_tensor(p) and p.numel() > 0:
                        tensor = p
                        break

            if tensor is None:
                return "0000"

            flat = tensor.flatten()
            values = []
            indices = [0, flat.shape[0] // 2, flat.shape[0] - 1]
            for i in indices:
                if i < flat.shape[0]:
                    values.append(flat[i].item())

            size = 0
            if hasattr(model_patcher, "model_size"):
                size = model_patcher.model_size()
            sample_str = f"{values}_{id(model_patcher):016x}_{size}"
            return hashlib.sha256(sample_str.encode()).hexdigest()[-4:]
        except Exception:
            return "err!"

    def _get_models_snapshot(self) -> List[Dict[str, Any]]:
        try:
            import comfy.model_management as model_management
        except Exception:
            return []

        snapshot: List[Dict[str, Any]] = []
        try:
            for loaded_model in model_management.current_loaded_models:
                model = loaded_model.model
                if model is None:
                    continue
                if str(getattr(loaded_model, "device", "")) != "cuda:0":
                    continue

                name = (
                    model.model.__class__.__name__
                    if hasattr(model, "model")
                    else type(model).__name__
                )
                model_hash = self._compute_model_hash(model)
                model_instance_id = getattr(model, "_instance_id", None)
                if model_instance_id is None:
                    model_instance_id = model_hash
                snapshot.append(
                    {
                        "name": str(name),
                        "id": str(model_instance_id),
                        "hash": str(model_hash or "????"),
                        "used": bool(getattr(loaded_model, "currently_used", False)),
                    }
                )
        except Exception:
            return []

        return snapshot

    def _update_model_context(self) -> None:
        snapshot = self._get_models_snapshot()
        selected = None

        used_models = [m for m in snapshot if m.get("used") and m.get("id")]
        if used_models:
            selected = used_models[-1]
        else:
            live_models = [m for m in snapshot if m.get("id")]
            if live_models:
                selected = live_models[-1]

        if selected is None:
            self._current_model_context = {
                "id": "unknown",
                "name": "unknown",
                "hash": "????",
            }
            return

        self._current_model_context = {
            "id": str(selected.get("id", "unknown")),
            "name": str(selected.get("name", "unknown")),
            "hash": str(selected.get("hash", "????") or "????"),
        }

    def scan(self, marker: str, refresh_model_context: bool = True) -> None:
        if not self._started or not _shm_debug_enabled():
            return

        if refresh_model_context:
            self._update_model_context()

        current = self._snapshot_shm()
        added = current - self._tracked_files
        removed = self._tracked_files - current
        self._tracked_files = current

        if not added and not removed:
            logger.debug("%s SHM:scan marker=%s changes=0", LOG_PREFIX, marker)
            return

        for filename in sorted(added):
            logger.info("%s SHM:created | %s", LOG_PREFIX, filename)
            model_id = self._current_model_context["id"]
            if model_id == "unknown":
                logger.error(
                    "%s SHM:model_association_missing | file=%s | reason=no_active_model_context",
                    LOG_PREFIX,
                    filename,
                )
            else:
                logger.info(
                    "%s SHM:model_association | model=%s | file=%s | name=%s | hash=%s",
                    LOG_PREFIX,
                    model_id,
                    filename,
                    self._current_model_context["name"],
                    self._current_model_context["hash"],
                )

        for filename in sorted(removed):
            logger.info("%s SHM:deleted | %s", LOG_PREFIX, filename)

        logger.debug(
            "%s SHM:scan marker=%s created=%d deleted=%d active=%d",
            LOG_PREFIX,
            marker,
            len(added),
            len(removed),
            len(self._tracked_files),
        )


_TRACKER = _SHMForensicsTracker()


def start_shm_forensics() -> None:
    _TRACKER.start()


def scan_shm_forensics(marker: str, refresh_model_context: bool = True) -> None:
    _TRACKER.scan(marker, refresh_model_context=refresh_model_context)


def stop_shm_forensics() -> None:
    _TRACKER.stop()


atexit.register(stop_shm_forensics)
