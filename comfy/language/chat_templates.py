from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path

KNOWN_CHAT_TEMPLATES = {}

logger = logging.getLogger(__name__)

def _update_known_chat_templates():
    try:
        _chat_templates = files(__package__) / "chat_templates"
        _extra_jinja_templates = {Path(traversable.name).stem: traversable.read_text().replace('    ', '').replace('\n', '') for traversable in _chat_templates.iterdir() if traversable.is_file()}
        KNOWN_CHAT_TEMPLATES.update(_extra_jinja_templates)
    except ImportError as exc:
        logger.warning("Could not load extra chat templates, some text models will fail", exc_info=exc)
